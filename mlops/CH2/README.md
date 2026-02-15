# CH2 Practice - Demo de MLOps + NLP (FastAPI + Qdrant + vLLM)

Demo mínima de **MLOps e NLP** com três etapas:

1. inserir textos e embeddings no Qdrant,
2. fazer busca semântica,
3. gerar resposta com LLM servido por vLLM.

## O que e vLLM?

**vLLM** e um servidor de inferência para LLMs. Ele expõe API compatível com o formato OpenAI e foi desenhado para tornar inferência de modelos mais eficiente.

Nesta prática, usamos o vLLM como serviço de geração de texto (sem fine-tuning).

## O que e Qdrant?

**Qdrant** e um banco vetorial. Ele armazena embeddings (vetores) e permite buscar os mais parecidos por similaridade semântica.

Nesta prática, ele guarda os embeddings dos textos e retorna contexto relevante para a resposta do LLM.

## Arquitetura (resumo)

- **api (FastAPI)**
  - Gera embeddings com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  - Escreve e consulta vetores no Qdrant.
  - Envia prompt para o vLLM.
- **qdrant**
  - Armazena vetores da coleção `workshop_docs`.
- **vllm**
  - Serve `microsoft/Phi-4-mini-4k-instruct` em API OpenAI-compatible.

Fluxo do endpoint `/ask`:
`pergunta -> embedding -> busca no Qdrant -> contexto -> prompt -> vLLM -> resposta`

## Estrutura

```text
mlops/CH2/practice
├── app
│   ├── main.py
│   ├── schemas.py
│   └── services.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Como rodar com Docker

1. Entre na pasta:

```bash
cd mlops/CH2/practice
```

1. Crie o arquivo de ambiente:

```bash
cp .env.example .env
```

1. Suba os serviços:

```bash
docker compose up --build
```

Serviços:

- API: `http://localhost:8001`
- Swagger: `http://localhost:8001/docs`
- Qdrant: `http://localhost:6333/dashboard`
- vLLM: `http://localhost:8000`

## Exemplos de requisição

### 1) Inserir textos

```bash
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "vLLM e um servidor de inferencia para LLMs.",
      "Qdrant e um banco vetorial para busca semantica.",
      "MLOps conecta desenvolvimento, deploy e operacao de modelos."
    ],
    "source": "workshop"
  }'
```

Resposta esperada:

```json
{"collection":"workshop_docs","inserted":3}
```

### 2) Busca semântica

```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query":"o que e banco vetorial?", "top_k":3}'
```

### 3) Perguntar com contexto (RAG simples)

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Explique vLLM e Qdrant em poucas linhas", "top_k":3}'
```

## Endpoints

- `POST /ingest`
  - Entrada: `texts` (lista), `source` (opcional)
  - Saída: coleção e quantidade inserida
- `POST /search`
  - Entrada: `query`, `top_k`
  - Saída: resultados similares com score
- `POST /ask`
  - Entrada: `question`, `top_k`
  - Saída: resposta do LLM + contexto recuperado
- `GET /health`
  - Healthcheck básico da API

## Materiais

- [Slides](https://www.canva.com/design/DAHBZXA8yew/Xiyd_hpJh1MYcXyYvrLF2Q/edit?utm_content=DAHBZXA8yew&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- [Documentação do vLLM](https://docs.vllm.ai/en/stable/)
- [Documentação do Qdrant](https://qdrant.tech/documentation/)
- [Documentação do FastAPI](https://fastapi.tiangolo.com/)
