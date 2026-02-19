# Cria o ambiente virtual chamado 'venv'
python -m venv venv

# Ativa o ambiente virtual (Linux / macOS)
source venv/bin/activate

# Ativa o ambiente virtual (Windows)
# venv\Scripts\activate

# Instala todas as dependências listadas no arquivo
pip install -r requirements.txt

# Baixa e roda o container do Qdrant na porta 6333. 
# A flag -d faz rodar em "detached mode" (em segundo plano) para não travar o seu terminal.
docker run -p 6333:6333 -d qdrant/qdrant

# Crie um arquivo .env na raiz do projeto e adicione sua chave:
OPENAI_API_KEY=sk-sua-chave-aqui

# Rodando o projeto

python src/ingestao.py -> Faz a ingestão do SQuADv2 no Qdrant

python src/metricas.py -> Roda uma avaliação nos 100 primeiros IDs do Dataset

python src/rag.py -> testa o rag via terminal

python src/api.py -> Sobe a nossa API

# Com a API rodando, você pode testar enviando uma requisição em outro terminal:

curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" -d '{"pergunta":"When did Beyonce start becoming popular?"}'