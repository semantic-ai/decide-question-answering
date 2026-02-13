"""
UC2 Stub - Subsidies RAG System
A minimal stub showing the flow for generic query → semantic search → LLM answer → response
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()


# Request/Response Models
class UC2Request(BaseModel):
    question: str
    dialog: Optional[List[dict]] = None
    filters: Optional[dict] = None
    top_n: Optional[int] = 5
    min_score: Optional[float] = 0.35


class SourceDoc(BaseModel):
    uri: str
    score: float
    title: Optional[str] = None


class UC2Response(BaseModel):
    answer: str
    sources: List[SourceDoc]


# Stub Functions
def semantic_search(query_text: str, filters: Optional[dict], top_n: int) -> List[dict]:
    """Stub: Semantic search for relevant decisions"""
    # TODO: Implement real semantic search
    mock_results = [
        {"uri": "http://example.org/decisions/123", "score": 0.82, "title": "Decision on renovation subsidy eligibility"},
        {"uri": "http://example.org/decisions/987", "score": 0.61, "title": "Decision on energy efficiency grants"},
        {"uri": "http://example.org/decisions/456", "score": 0.45, "title": "Decision on home improvement subsidies"},
    ]
    return mock_results[:top_n]


def apply_relevance_threshold(docs: List[dict], min_score: float) -> List[dict]:
    """Stub: Filter documents by relevance threshold"""
    # TODO: Implement real threshold filtering
    return [doc for doc in docs if doc.get("score", 0) >= min_score]


def generate_answer(question: str, retrieved_docs: List[dict]) -> str:
    """Stub: Generate answer using LLM with retrieved documents"""
    # TODO: Implement real LLM generation
    doc_count = len(retrieved_docs)
    return f"STUB: Based on {doc_count} retrieved decisions, here is a placeholder answer to: {question}"


# Orchestration
def process_uc2_request(request: UC2Request) -> UC2Response:
    """Main UC2 pipeline: question → search → LLM → response"""
    # Step 1: Semantic search
    filters = request.filters or {}
    retrieved_docs = semantic_search(request.question, filters, request.top_n or 5)
    
    # Step 2: Apply relevance threshold
    min_score = request.min_score or 0.35
    filtered_docs = apply_relevance_threshold(retrieved_docs, min_score)
    
    # Step 3: Generate answer with LLM
    answer = generate_answer(request.question, filtered_docs)
    
    # Step 4: Format response
    sources = [
        SourceDoc(uri=doc["uri"], score=doc["score"], title=doc.get("title"))
        for doc in filtered_docs
    ]
    
    return UC2Response(answer=answer, sources=sources)


# FastAPI Endpoint
@router.post("/uc2/answer", response_model=UC2Response)
async def uc2_answer_endpoint(request: UC2Request):
    """UC2 endpoint: Accepts question/dialog, returns answer + source URIs"""
    return process_uc2_request(request)
