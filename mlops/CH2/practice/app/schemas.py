from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, description="Lista de textos para indexar"
    )
    source: str = Field(default="manual", min_length=1, description="Origem dos textos")


class IngestResponse(BaseModel):
    collection: str
    inserted: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class SearchResult(BaseModel):
    text: str
    score: float
    source: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    context: list[SearchResult]
