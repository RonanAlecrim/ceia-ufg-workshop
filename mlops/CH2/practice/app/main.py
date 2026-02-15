import logging
import os

import requests
from fastapi import FastAPI, HTTPException
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from services import DemoService

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

app = FastAPI(title="CH2 MLOps + NLP Demo", version="1.0.0")
service = DemoService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    cleaned = [text.strip() for text in request.texts if text.strip()]
    if not cleaned:
        raise HTTPException(
            status_code=400, detail="A lista de textos está vazia após limpeza."
        )

    try:
        inserted = service.ingest(texts=cleaned, source=request.source)
        return IngestResponse(collection=service.collection_name, inserted=inserted)
    except (ResponseHandlingException, UnexpectedResponse) as exc:
        raise HTTPException(
            status_code=503, detail=f"Qdrant indisponível: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erro ao ingerir textos: {exc}"
        ) from exc


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        results = service.semantic_search(query=request.query, top_k=request.top_k)
        if not results:
            return SearchResponse(results=[])
        return SearchResponse(results=results)
    except (ResponseHandlingException, UnexpectedResponse) as exc:
        raise HTTPException(
            status_code=503, detail=f"Qdrant indisponível: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erro na busca semântica: {exc}"
        ) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        answer, context = service.ask(question=request.question, top_k=request.top_k)
        return AskResponse(answer=answer, context=context)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503, detail=f"vLLM indisponível: {exc}"
        ) from exc
    except (ResponseHandlingException, UnexpectedResponse) as exc:
        raise HTTPException(
            status_code=503, detail=f"Qdrant indisponível: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Erro ao gerar resposta: {exc}"
        ) from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVER_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
