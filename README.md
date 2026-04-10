## UC2 Subsidies RAG System

This service implements the HTTP flow for a subsidies RAG (Retrieval-Augmented Generation) system.

The flow is based on the following:
- Create an HTTP service that accepts a question in JSON format.
- Call the embedding service to obtain an embedding for the question.
- Use the embedding to perform a semantic search and return the top retrieved decisions together with their URIs.
- Resolve the titles and content of the retrieved decisions from the SPARQL endpoint.
- Pass the question plus the retrieved documents to an LLM to generate a response.

*Note: Relevance scoring and threshold filtering are not yet implemented as the retrieval API does not return per-document scores. The current default is to return the first 3 retrieved documents.*

### Disclaimer

This project currently uses direct service-to-service communication. This is a temporary choice while we determine how these AI services should fit within the LBLod paradigm. This approach should be treated as project-specific and should not be copied as a general pattern for other applications.

### LLM provider configuration

The generation step uses LangChain's [`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/) to allow switching providers without any code changes:

| Variable | Description | Default |
|---|---|---|
| `GENERATION_PROVIDER` | LangChain provider name (e.g. `ollama`, `mistralai`, `openai`) | `ollama` |
| `GENERATION_MODEL` | Model name for the selected provider | `mistral-nemo` |
| `GENERATION_ENDPOINT` | Base URL for self-hosted providers (e.g. Ollama) | — |
| `GENERATION_API_KEY` | API key for cloud providers | — |

To switch providers, change `GENERATION_PROVIDER` and `GENERATION_MODEL` and install the matching `langchain-<provider>` package in `requirements.txt`. No code changes needed.

### Setup

```bash
docker compose -f docker-compose.debug.yml up --build
```

### Enrichment query configuration

The service reads its enrichment SPARQL query from:

- `/config/enrichment-query.rq`

The debug compose file mounts the local `config` folder to `/config`, so this file is picked up automatically during local development.

For reuse in other apps, mount an app-specific folder to `/config` (for example `./config/question-answering/:/config`) and provide an `enrichment-query.rq` in that folder. This allows changing enrichment behavior (for example from "decisions" to another domain) without changing service code.

### Verification

```bash
curl -X POST http://localhost:8000/uc2/answer -H "Content-Type: application/json" -d "{\"question\": \"What subsidies exist for renovating an older home?\"}"
```

```bash
curl -X POST http://localhost:8000/uc2/answer -H "Content-Type: application/json" -d "{\"question\": \"Als ik iets aan mijn huis verbouw, ben ik dan zelf verantwoordelijk voor beschadigingen aan de inrichting van het openbaar domein, groenaanleg, bermen, trottoirs, boordstenen, straatkolken en de rijweg die te wijten zijn aan de bouwactiviteit ?\"}"
```

### Expected input

```json
{
  "question": "What subsidies exist for renovating an older home?",
  "top_n": 3
}
```

- `question`: The current user question being asked
- `top_n`: Optional number of retrieved documents to include, defaults to `3`

### Expected output

```json
{
  "answer": "Based on the retrieved documents, ...",
  "sources": [
    {
      "uri": "https://example.org/document/1",
      "title": "Example title",
      "content": "Document content text..."
    }
  ]
}
```

- `answer`: The generated answer from the LLM.
- `sources`: The retrieved source documents used for the answer.
  - `uri`: The document identifier returned by the retrieval API.
  - `title`: The document title resolved from the SPARQL endpoint.
  - `content`: The document content resolved from the SPARQL endpoint.

### Possible improvements

- **Relevance scores from retrieval API**: The search API currently does not return similarity scores. If scores were available, we could filter out low-relevance documents before passing them to the LLM, reducing noise and improving answer quality.
- **Cross-encoder reranking**: Add a reranking step between retrieval and generation. A cross-encoder is a small model that takes a question and a document together as input and outputs a relevance score. Unlike embeddings (which compress text into vectors separately and then compare), a cross-encoder reads both texts side by side, so it catches nuances that embeddings miss. The trade-off is that it's slower (it must run once per document), which is why it's used as a second stage: fast retrieval narrows down candidates, then the cross-encoder re-scores and filters them before passing to the LLM.