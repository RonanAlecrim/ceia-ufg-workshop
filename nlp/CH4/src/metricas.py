from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np

def calcular_metricas(rankings, k=5):
    """
    Calcula Recall@k, Precision@k, MRR e NDCG.
    rankings: lista de booleanos indicando se o doc na posição 'i' é relevante.
    """
    relevantes_no_top_k = sum(rankings[:k])
    
    # Recall@k (Temos apenas 1 documento perfeitamente relevante por pergunta no SQuAD)
    recall = 1.0 if relevantes_no_top_k > 0 else 0.0
    
    # Precision@k
    precision = relevantes_no_top_k / k
    
    # MRR e NDCG
    mrr = 0.0
    ndcg = 0.0
    for i, is_relevant in enumerate(rankings[:k]):
        if is_relevant:
            rank = i + 1
            if mrr == 0.0: # Pega o rank do primeiro relevante encontrado
                mrr = 1.0 / rank
            ndcg = 1.0 / np.log2(rank + 1) # NDCG simplificado para relevância binária (1 doc correto)
            break 
            
    return recall, precision, mrr, ndcg

def main():
    print("Iniciando avaliação do motor de busca...")
    dataset = load_dataset("squad_v2", split="train[:100]") # Pegamos as 100 primeiras perguntas para avaliar
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(host="localhost", port=6333)
    
    k = 5
    metricas = {"recall": [], "precision": [], "mrr": [], "ndcg": []}
    
    for item in dataset:
        pergunta = item["question"]
        contexto_verdadeiro = item["context"] # O gabarito
        
        # 1. Gerar embedding da pergunta e buscar
        vetor_pergunta = model.encode(pergunta).tolist()
        resultados = client.search(
            collection_name="squad_docs",
            query_vector=vetor_pergunta,
            limit=k
        )
        
        # 2. Verificar se o contexto retornado bate com o verdadeiro
        # Cria uma lista [True, False, False...] baseada nos retornos
        rankings = [res.payload["texto"] == contexto_verdadeiro for res in resultados]
        
        # 3. Calcular
        r, p, m, n = calcular_metricas(rankings, k)
        metricas["recall"].append(r)
        metricas["precision"].append(p)
        metricas["mrr"].append(m)
        metricas["ndcg"].append(n)
        
    print(f"\n--- Resultados da Avaliação (Top-{k}) ---")
    print(f"Recall@{k}:    {np.mean(metricas['recall']):.4f}")
    print(f"Precision@{k}: {np.mean(metricas['precision']):.4f}")
    print(f"MRR:         {np.mean(metricas['mrr']):.4f}")
    print(f"NDCG@{k}:      {np.mean(metricas['ndcg']):.4f}")

if __name__ == "__main__":
    main()