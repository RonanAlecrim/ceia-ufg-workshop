import logging
import os
import uuid

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer


class DemoService:
    def __init__(self) -> None:
        self.collection_name = os.getenv("QDRANT_COLLECTION", "workshop_docs")
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.vllm_url = os.getenv("VLLM_URL", "http://vllm:8000/v1/chat/completions")
        self.vllm_model = os.getenv("VLLM_MODEL", "microsoft/Phi-4-mini-4k-instruct")

        qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        logging.info("Loading embedding model: %s", self.embedding_model_name)
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        logging.info("Connecting to Qdrant: %s:%s", qdrant_host, qdrant_port)
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

    def ensure_collection(self) -> None:
        if not self.qdrant.collection_exists(self.collection_name):
            logging.info("Creating collection: %s", self.collection_name)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            return

        collection = self.qdrant.get_collection(self.collection_name)
        current_size = collection.config.params.vectors.size
        if current_size != self.embedding_dim:
            raise ValueError(
                f"Collection '{self.collection_name}' uses vector size {current_size}, "
                f"but embedding model produces size {self.embedding_dim}."
            )

    def ingest(self, texts: list[str], source: str) -> int:
        logging.info("embed stage: ingest %s texts", len(texts))
        self.ensure_collection()

        vectors = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, "source": source},
            )
            for text, vector in zip(texts, vectors)
        ]

        logging.info("search stage: upsert %s points", len(points))
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def semantic_search(self, query: str, top_k: int) -> list[dict]:
        self.ensure_collection()

        count_result = self.qdrant.count(
            collection_name=self.collection_name, exact=True
        )
        if count_result.count == 0:
            return []

        logging.info("embed stage: search query")
        query_vector = self.embedder.encode([query], normalize_embeddings=True)[
            0
        ].tolist()

        logging.info("search stage: top_k=%s", top_k)
        hits = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("source", "unknown"),
                "score": float(hit.score),
            }
            for hit in hits
        ]

    def ask(self, question: str, top_k: int) -> tuple[str, list[dict]]:
        context = self.semantic_search(question, top_k)

        if not context:
            return (
                "Ainda não há textos indexados. Use /ingest para inserir conteúdo antes de perguntar.",
                [],
            )

        context_text = "\n".join([f"- {item['text']}" for item in context])
        prompt = (
            "Você é um assistente didático para uma aula de MLOps e NLP. "
            "Responda em português de forma curta e objetiva.\n\n"
            f"Contexto:\n{context_text}\n\n"
            f"Pergunta: {question}"
        )

        payload = {
            "model": self.vllm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Responda apenas com base no contexto quando possível.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
        }

        logging.info("generate stage: calling vLLM")
        response = requests.post(self.vllm_url, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer, context
