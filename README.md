## UC2 Subsidies RAG System

This service implements the HTTP flow for a subsidies RAG (Retrieval-Augmented Generation) system.

The flow is based on the following:
- Create an HTTP service that accepts a question in JSON format.
- Call the embedding service to obtain an embedding for the question.
- Use the embedding to perform a semantic search and return the top retrieved decisions together with their URIs.
- Resolve the titles of the retrieved decisions from the SPARQL endpoint.
- Pass the question plus the retrieved decisions to an LLM to generate a response (currently a stub).

*Note: Relevance scoring and threshold filtering are not yet implemented as the retrieval API does not return per-document scores. The current default is to return the first 3 retrieved documents.*

### Disclaimer

This project currently uses direct service-to-service communication. This is a temporary choice while we determine how these AI services should fit within the LBLod paradigm. This approach should be treated as project-specific and should not be copied as a general pattern for other applications.

### Setup

```bash
docker-compose up --build
```

### Verification

```bash
curl -X POST http://localhost:8000/uc2/answer -H "Content-Type: application/json" -d "{\"question\": \"What subsidies exist for renovating an older home?\"}"
```

**Format notes:**
- `question`: The current user question being asked
- `top_n`: Optional number of retrieved documents to include, defaults to `3`

### Expected input

```json
{
  "question": "What subsidies exist for renovating an older home?",
  "top_n": 3
}
```

### Expected output

```json
{
  "answer": "STUB: Based on 3 retrieved decisions, here is a placeholder answer to: What subsidies exist for renovating an older home?",
  "sources": [
    {
      "uri": "https://example.org/document/1",
      "title": "Example title"
    }
  ]
}
```

The response contains:
- `answer`: The generated answer text. This is currently a stub response.
- `sources`: The retrieved source documents used for the answer.
- `uri`: The document identifier returned by the retrieval API, exposed by this service as `uri`.
- `title`: The document title resolved from the SPARQL endpoint when available.
