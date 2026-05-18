"""
Question Answering RAG System
Flow: question → embedding API → semantic search → top N decisions → response
"""

import os
import requests
from fastapi import APIRouter
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from escape_helpers import sparql_escape_uri
from helpers import query
from langchain.chat_models import init_chat_model

router = APIRouter()

SEARCH_API_URL = os.environ.get("SEARCH_API_URL")
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL")
GENERATION_ENDPOINT = os.environ.get("GENERATION_ENDPOINT")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "mistral-nemo")
GENERATION_PROVIDER = os.environ.get("GENERATION_PROVIDER", "ollama")
GENERATION_API_KEY = os.environ.get("GENERATION_API_KEY")
GENERATION_TIMEOUT = float(os.environ.get("GENERATION_TIMEOUT", "300.0"))
MAX_CONTENT_CHARS = int(os.environ.get("MAX_CONTENT_CHARS", "1000"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "10.0"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.72"))
EMBEDDING_K = int(os.environ.get("EMBEDDING_K", "10"))
EMBEDDING_NUM_CANDIDATES = int(os.environ.get("EMBEDDING_NUM_CANDIDATES", "400"))

DEFAULT_ENRICHMENT_SPARQL_TEMPLATE = """
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX epvoc: <https://data.europarl.europa.eu/def/epvoc#>
SELECT ?s ?title ?content
WHERE {
  VALUES ?s { {{values}} }
  ?s eli:title ?title .
  OPTIONAL { ?s epvoc:expressionContent ?content . }
}
"""
ENRICHMENT_SPARQL_TEMPLATE_FILE = "/config/enrichment-query.rq"

try:
    with open(ENRICHMENT_SPARQL_TEMPLATE_FILE, encoding="utf-8") as query_file:
        ENRICHMENT_SPARQL_TEMPLATE = query_file.read()
except OSError:
    # TODO: Log this error instead of printing it
    print("Warning: Could not load enrichment SPARQL template from file. Using default template.")
    ENRICHMENT_SPARQL_TEMPLATE = DEFAULT_ENRICHMENT_SPARQL_TEMPLATE


# Request/Response Models
class AnswerRequest(BaseModel):
    question: str = Field(..., description="The question to answer.", examples=["Welke subsidies zijn er voor dakisolatie?"])
    top_n: Optional[int] = Field(5, description="Number of source documents to retrieve.", ge=1, le=20)
    localAuthority: Optional[str] = Field(None, description="URI of the local authority to filter by.")


class SourceDoc(BaseModel):
    uri: str = Field(..., description="URI of the source document.")
    title: Optional[str] = Field(None, description="Document title.")
    content: Optional[str] = Field(None, description="Relevant excerpt used to generate the answer.")
    score: Optional[float] = Field(None, description="Similarity score from semantic search.")


class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Answer generated from the retrieved documents, in the same language as the question.")
    sources: List[SourceDoc] = Field(..., description="Documents used to generate the answer.")


def embed_question(question: str) -> List[float]:
    """Call the embedding service to obtain an embedding for the question.

    Args:
        question (str): The user question to embed.

    Returns:
        List[float]: The embedding vector returned by the embedding service.
    """
    response = requests.post(
        EMBEDDING_API_URL,
        json={"input": question},
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("embedding", [])


def semantic_search(question: str, top_n: int, local_authority: Optional[str] = None) -> List[SourceDoc]:
    """Perform semantic search and return normalized source documents.

    Args:
        question (str): The user question used for retrieval.
        top_n (int): The maximum number of documents to return.
        local_authority (str, optional): URI of the local authority to filter results by.

    Returns:
        List[SourceDoc]: The top retrieved documents, normalized to use `uri`
        as the internal identifier.
    """
    embedding = embed_question(question)
    embedding_string = ",".join(str(value) for value in embedding)
    filter_params = {
        ":embedding:description-vector": f"{EMBEDDING_K}:{EMBEDDING_NUM_CANDIDATES}:{embedding_string}",
    }
    if local_authority:
        filter_params["owning-body"] = local_authority
    payload = {"filter": filter_params}
    response = requests.post(
        SEARCH_API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    results = normalize_search_results(data.get("data", []))
    results = [doc for doc in results if doc.score is None or doc.score >= MIN_SCORE]
    return results[:top_n]


def fetch_documents(sources: List[SourceDoc]) -> List[SourceDoc]:
    """Fetch document metadata from Virtuoso and enrich the source documents.

    Args:
        sources (List[SourceDoc]): The normalized source documents to enrich.

    Returns:
        List[SourceDoc]: The input source documents enriched with metadata
        retrieved from the SPARQL endpoint.
    """
    if not sources:
        return []

    uris = [source.uri for source in sources]
    values = " ".join(sparql_escape_uri(uri) for uri in uris)
    sparql_query = ENRICHMENT_SPARQL_TEMPLATE.replace("{{values}}", values)

    data = query(sparql_query)

    doc_map: dict[str, dict] = {}
    for binding in data.get("results", {}).get("bindings", []):
        subject = binding.get("s", {}).get("value")
        title = binding.get("title", {}).get("value")
        content = binding.get("content", {}).get("value")
        if subject and subject not in doc_map:
            doc_map[subject] = {"title": title, "content": " ".join(content.split()) if content else content}

    return [
        SourceDoc(uri=source.uri, score=source.score, **doc_map.get(source.uri, {}))
        for source in sources
    ]



def normalize_search_results(docs: List[dict]) -> List[SourceDoc]:
    """Normalize retrieval API results to internal source documents.

    Args:
        docs (List[dict]): The raw documents returned by the retrieval API.

    Returns:
        List[SourceDoc]: The normalized source documents. The retrieval API
        returns document identifiers as `id`, but within this service we
        expose and work with them as `uri`.
    """
    return [
        SourceDoc(uri=doc["id"], score=doc.get("score"))
        for doc in docs if doc.get("attributes").get("uri")
    ]

def _get_llm():
    """Instantiate the LLM for the configured provider. Returns a BaseChatModel.
    Uses LangChain's init_chat_model with a 'provider:model' string — no provider-specific
    branching needed. Switching providers only requires changing GENERATION_PROVIDER and
    installing the matching langchain-<provider> package.
    """
    return init_chat_model(
        f"{GENERATION_PROVIDER}:{GENERATION_MODEL}",
        base_url=GENERATION_ENDPOINT,
        api_key=GENERATION_API_KEY,
        timeout=GENERATION_TIMEOUT,
    )


def generate_answer(question: str, retrieved_docs: List[SourceDoc]) -> str:
    """Generate an answer using the LLM with retrieved documents as context."""
    doc_blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        title = doc.title or doc.uri
        content = (doc.content or "")[:MAX_CONTENT_CHARS]
        doc_blocks.append(f"Document {i}\nTitle: {title}\nContent: {content}")

    context = "\n\n".join(doc_blocks)
    prompt = (
        "You are a helpful assistant answering questions about city council decisions.\n\n"
        "Below are retrieved documents that may or may not be relevant to the question. "
        "Use only the documents that are actually relevant to answer the question. "
        "If none of the documents are relevant, say so.\n"
        "Answer ONLY based on the provided documents. Do not use outside knowledge.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "IMPORTANT: You MUST answer in the exact same language as the question above. "
        "Do NOT answer in English unless the question is in English.\n\n"
        "Answer:"
    )
    result = _get_llm().invoke(prompt)
    return result.content


# Orchestration
def process_request(request: AnswerRequest) -> AnswerResponse:
    """Main pipeline: question → search → LLM → response"""
    sources = semantic_search(request.question, request.top_n or 5, request.localAuthority)
    if not sources:
        return AnswerResponse(answer="No relevant documents were found to answer this question.", sources=[])
    sources = fetch_documents(sources)
    answer = generate_answer(request.question, sources)
    return AnswerResponse(answer=answer, sources=sources)


# FastAPI Endpoint
@router.get("/question-answering/documentation", include_in_schema=False)
def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/question-answering/openapi.json",
        title="Question Answering API",
    )


@router.post(
    "/question-answering/answer",
    response_model=AnswerResponse,
    summary="Answer a question",
    description="Finds relevant local authority decisions using semantic search and returns an LLM-generated answer with source documents.",
)
def answer_endpoint(request: AnswerRequest):
    return process_request(request)
