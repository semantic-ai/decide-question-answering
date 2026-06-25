"""
Question Answering RAG System
Flow: question → embedding API → semantic search → top N decisions → response
"""

import os
import json
import time
import requests
from fastapi import APIRouter
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from escape_helpers import sparql_escape_uri, sparql_escape_string, sparql_escape_datetime, sparql_escape
from helpers import query, generate_uuid, update
from langchain.chat_models import init_chat_model
from datetime import datetime, timezone

router = APIRouter()

SEARCH_API_URL = os.environ.get("SEARCH_API_URL", "http://search/expressions/search")
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL")
GENERATION_ENDPOINT = os.environ.get("GENERATION_ENDPOINT")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "mistral-nemo")
GENERATION_PROVIDER = os.environ.get("GENERATION_PROVIDER", "ollama")
GENERATION_API_KEY = os.environ.get("GENERATION_API_KEY")
GENERATION_TIMEOUT = float(os.environ.get("GENERATION_TIMEOUT", "300.0"))
OLLAMA_PULL_TIMEOUT = float(os.environ.get("OLLAMA_PULL_TIMEOUT", "1800.0"))
OLLAMA_STARTUP_WAIT = float(os.environ.get("OLLAMA_STARTUP_WAIT", "60.0"))
MAX_CONTENT_CHARS = int(os.environ.get("MAX_CONTENT_CHARS", "50000"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "10.0"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.72"))
EMBEDDING_K = int(os.environ.get("EMBEDDING_K", "30"))
EMBEDDING_NUM_CANDIDATES = int(os.environ.get("EMBEDDING_NUM_CANDIDATES", "100"))
TITLE_FALLBACK_CHARS = int(os.environ.get("TITLE_FALLBACK_CHARS", "80"))

QUESTION_BASE_URI = "http://data.lblod.info/id/questions/"
ANSWER_BASE_URI   = "http://data.lblod.info/id/answers/"
QUOTATION_BASE_URI = "http://data.lblod.info/id/quotations/"

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
    answer_id: Optional[str] = Field(None, description="uuid of the answer.")
    question_id: str = Field(..., description="uuid of the question.")
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
    """Perform semantic search and return source documents.

    Sends a pre-filtered kNN as a raw Elasticsearch query to mu-search's
    `/:type/search` endpoint (so mu-auth / allowed-groups still apply and we don't
    hardcode the index). When a local authority URI is provided, its `owning-body`
    term sits in the bool `filter` next to the `knn` in `must`, so Elasticsearch
    pre-filters to that authority's documents and a small `k` suffices. The resource
    URI is `attributes.uri`; the similarity score is the doc-level `score`.
    """
    embedding = embed_question(question)
    bool_query = {
        "must": [{
            "knn": {
                "field": "description-vector",
                "query_vector": embedding,
                "k": EMBEDDING_K,
                "num_candidates": EMBEDDING_NUM_CANDIDATES,
            }
        }]
    }
    if local_authority:
        bool_query["filter"] = [{"term": {"owning-body": local_authority}}]
    body = {"query": {"bool": bool_query}, "size": top_n}
    response = requests.post(
        SEARCH_API_URL,
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    results = [
        SourceDoc(uri=doc["attributes"]["uri"], score=doc.get("score"))
        for doc in data if doc.get("attributes", {}).get("uri")
    ]
    results = [doc for doc in results if doc.score is None or doc.score >= MIN_SCORE]
    return results[:top_n]


def _derive_title(title: Optional[str], content: Optional[str]) -> Optional[str]:
    """Return a display title for a source document.

    Prefers an actual title (direct ``eli:title`` or one supplied via an
    annotation). When none is available — e.g. Bamberg expressions without a
    title annotation, or one with an empty value — it falls back to the first
    ``TITLE_FALLBACK_CHARS`` characters of the content, with markdown markers
    and redundant whitespace stripped. Returns ``None`` if neither exists.
    """
    if title and title.strip():
        return title.strip()
    if content and content.strip():
        snippet = " ".join(content.split()).lstrip("*# ").strip()
        return snippet[:TITLE_FALLBACK_CHARS] or None
    return None


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
        if not subject:
            continue
        title = binding.get("title", {}).get("value")
        content = binding.get("content", {}).get("value")
        entry = doc_map.setdefault(subject, {"title": None, "content": None})
        if content and not entry["content"]:
            entry["content"] = " ".join(content.split())
        if title and title.strip() and not (entry["title"] and entry["title"].strip()):
            entry["title"] = title.strip()

    # Fall back to a content snippet when no usable title was found.
    for entry in doc_map.values():
        entry["title"] = _derive_title(entry["title"], entry["content"])

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
        SourceDoc(uri=doc["attributes"]["uri"], score=doc.get("score"))
        for doc in docs if doc.get("attributes", {}).get("uri")
    ]

def _ensure_ollama_model() -> None:
    """Pull GENERATION_MODEL into Ollama at startup if it isn't already present.

    Ollama's chat endpoint won't pull a missing model on demand (it 404s), so we
    check ``/api/show`` and, if absent, ``/api/pull``. Blocks until ready and raises
    on failure, so the service refuses to start with an unavailable model rather than
    failing on the first request. Waits up to OLLAMA_STARTUP_WAIT seconds for Ollama
    to be reachable (it may still be booting). Assumes the provider is Ollama and
    GENERATION_ENDPOINT is set — see the guarded startup call below.
    """
    base = GENERATION_ENDPOINT.rstrip("/")

    # Ollama may not be up yet at startup — wait (bounded) for it to answer before
    # treating an error as fatal, so a boot-order race doesn't crash the service.
    deadline = time.monotonic() + OLLAMA_STARTUP_WAIT
    while True:
        try:
            show = requests.post(
                f"{base}/api/show", json={"model": GENERATION_MODEL}, timeout=REQUEST_TIMEOUT
            )
            break
        except requests.RequestException as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Ollama at {base} not reachable after {OLLAMA_STARTUP_WAIT}s: {exc}"
                ) from exc
            print(f"Waiting for Ollama at {base} to become available...", flush=True)
            time.sleep(3)

    if show.status_code == 200:
        print(f"Ollama model '{GENERATION_MODEL}' already present.", flush=True)
        return
    if show.status_code != 404:
        show.raise_for_status()

    # Model is absent — pull it. /api/pull streams NDJSON status lines.
    print(f"Ollama model '{GENERATION_MODEL}' not found; pulling it (this may take a while)...", flush=True)
    with requests.post(
        f"{base}/api/pull",
        json={"model": GENERATION_MODEL},
        stream=True,
        timeout=OLLAMA_PULL_TIMEOUT,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            status = json.loads(line)
            if status.get("error"):
                raise RuntimeError(
                    f"Failed to pull Ollama model '{GENERATION_MODEL}': {status['error']}"
                )
    print(f"Ollama model '{GENERATION_MODEL}' is ready.", flush=True)


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

def store_question(request: AnswerRequest) -> str:
    """Persist a schema:Question to the triplestore and return the uuid for the question."""
    question_uuid = generate_uuid()
    question_uri  = f"{QUESTION_BASE_URI}{question_uuid}"
    created = datetime.now(timezone.utc)
    triples = f"""
        <{question_uri}> a schema:Question ;
            mu:uuid       {sparql_escape_string(question_uuid)}
            dct:created   {sparql_escape_datetime(created)} ;
            schema:text   {sparql_escape_string(request.question)} .
    """
    if request.localAuthority:
        triples += f"\n        <{question_uri}> ext:owningBody {sparql_escape_uri(request.localAuthority)} ."

    update(f"""
        PREFIX schema: <http://schema.org/>
        PREFIX dct:    <http://purl.org/dc/terms/>
        PREFIX ext:    <http://mu.semte.ch/vocabularies/ext/>
        PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>

        INSERT DATA {{
          {triples}
        }}
    """)
    return question_uuid

def store_question_prompt(question_uuid: str, prompt: str):
    """Persist the prompt in a known schema:Question."""
    question_uri  = f"{QUESTION_BASE_URI}{question_uuid}"
    triples = f"""
        <{question_uri}> dct:description {sparql_escape_string(prompt)} .
    """
    update(f"""
        PREFIX dct:    <http://purl.org/dc/terms/>

        INSERT DATA {{
          {triples}
        }}
    """)

def store_question_answer(question_uuid: str, answer: str, sources: List[SourceDoc]) -> str:
    """Persist the answer to a known schema:Question in the triplestore, and return the answer uuid."""
    question_uri  = f"{QUESTION_BASE_URI}{question_uuid}"
    created = datetime.now(timezone.utc)
    answer_uuid = generate_uuid()
    answer_uri  = f"{ANSWER_BASE_URI}{answer_uuid}"
    llm_uri     = f"urn:llm:{GENERATION_PROVIDER}:{GENERATION_MODEL}"
    triples = f"""
        <{answer_uri}> a schema:Answer ;
            mu:uuid       {sparql_escape_string(answer_uuid)}
            dct:created   {sparql_escape_datetime(created)} ;
            schema:text   {sparql_escape_string(answer)} ;
            dct:creator  <{llm_uri}> .
        <{question_uri}> schema:suggestedAnswer <{answer_uri}> .
    """

    for source in sources:
        quotation_uuid = generate_uuid()
        quotation_uri  = f"{QUOTATION_BASE_URI}{quotation_uuid}"
        triples += f"\n        <{quotation_uri}> a schema:Quotation ;"
        triples += f"\n            oa:hasSource {sparql_escape_uri(source.uri)} ;"
        if source.score is not None:
            triples += f"\n            ext:confidence {sparql_escape(source.score)} ."
        triples += f"\n        <{answer_uri}> schema:citation <{quotation_uri}> ."

    update(f"""
        PREFIX schema: <http://schema.org/>
        PREFIX dct:    <http://purl.org/dc/terms/>
        PREFIX ext:    <http://mu.semte.ch/vocabularies/ext/>
        PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
        PREFIX oa:     <http://www.w3.org/ns/oa#>
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>

        INSERT DATA {{
          {triples}
        }}
    """)
    return answer_uuid



def generate_answer(question: str, retrieved_docs: List[SourceDoc], question_id: str) -> str:
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
    store_question_prompt(question_id, prompt)
    result = _get_llm().invoke(prompt)
    return result.content


# Orchestration
def process_request(request: AnswerRequest) -> AnswerResponse:
    """Main pipeline: question → search → LLM → response"""

    question_id = store_question(request)
    sources = semantic_search(request.question, request.top_n or 5, request.localAuthority)
    if not sources: # when no relevant documents were found, no answer is generated, but the question is still stored
        return AnswerResponse(question_id=question_id, answer="No relevant documents were found to answer this question.", sources=[])
    sources = fetch_documents(sources)
    answer = generate_answer(request.question, sources, question_id)
    answer_id = store_question_answer(question_id, answer, sources)
    return AnswerResponse(question_id=question_id, answer_id=answer_id, answer=answer, sources=sources)


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

if GENERATION_PROVIDER == "ollama" and GENERATION_ENDPOINT:
    _ensure_ollama_model()
