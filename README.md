## Local decision question answering RAG System

This service implements the HTTP flow for a local decision question answering RAG (Retrieval-Augmented Generation) system.

The flow is based on the following:
- Create an HTTP service that accepts a question in JSON format.
- Call the embedding service to obtain an embedding for the question (cached per normalized question).
- Retrieve candidate decisions with a hybrid search: a pre-filtered kNN leg via mu-search's raw-DSL `/:type/search` endpoint (the `owning-body`/city filter sits in the `bool` `filter` next to the `knn`, so Elasticsearch pre-filters to that city) and a fuzzy BM25 leg via the `/:type/search` full-text endpoint (`multi_match` with `fuzziness: AUTO`), fused with Reciprocal Rank Fusion.
- Re-rank the fused pool with a cross-encoder model baked into the container image.
- Resolve the titles and content of the retrieved decisions from the SPARQL endpoint.
- Pass the question plus the retrieved documents to an LLM to generate a response.

### Hybrid retrieval & reranking

Pure dense kNN retrieval is brittle to spelling mistakes: a single typo in the
question ("sbusidies") shifts the embedding far enough that a completely
different set of documents tops the list. The retrieval stage therefore combines
two legs:

1. **Dense leg** — the kNN search described above (up to 10 candidates; the
   endpoint caps results at 10).
2. **Lexical leg** — a fuzzy BM25 `multi_match` over the `description` fields
   (`fuzziness: AUTO`), which anchors near-miss spellings to the documents that
   use the correct term.

The legs are fused with Reciprocal Rank Fusion (`RRF_K`), duplicates (identical
description text) are collapsed, dense scores are still filtered by `MIN_SCORE`,
and a multilingual cross-encoder (`BAAI/bge-reranker-v2-m3`) re-scores the fused
pool against the question. The cross-encoder is loaded at startup from
`RERANK_MODEL_PATH` and is baked into the image at build time (pinned model
revision, see `Dockerfile`), so containers start without any runtime downloads.
Every step is fail-soft: a failed leg, a missing model, or a failed prediction
falls back to the remaining retrieval order instead of failing the request.

Note: the lexical leg is a global full-text search — the mu-search endpoint does
not accept a filter clause — so its results are filtered by `owning-body` in
this service. In a multi-authority corpus the documents of one authority rank
deep in the global BM25 results, which is why `LEXICAL_SIZE` defaults to 200.

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

**Default — Mistral cloud** The API key is read from a `.env`

```yaml
GENERATION_PROVIDER: "mistralai"
GENERATION_MODEL: "mistral-medium-3-5"
GENERATION_API_KEY: ${GENERATION_API_KEY}
```

**To use a local Ollama model instead**, drop `GENERATION_API_KEY` and point at the Ollama endpoint:

```yaml
GENERATION_PROVIDER: "ollama"
GENERATION_ENDPOINT: "http://ollama:11434"
GENERATION_MODEL: "mistral-nemo"
```

When the provider is `ollama`, the service automatically pulls `GENERATION_MODEL` **at startup** if it isn't already present in Ollama (it checks `/api/show` and, if missing, calls `/api/pull`). So you can point `GENERATION_MODEL` at any model from the [Ollama library](https://ollama.com/library), e.g. `gemma4:12b`, without pre-pulling it.

This is fail-fast: the service does not finish starting until the model is available, and if the model can't be pulled (it doesn't exist, the disk is full, etc.) the container exits instead of starting. The pull can take a while for large models — tune `OLLAMA_PULL_TIMEOUT` if needed. Because the `ollama` container may still be booting when this service starts, startup waits up to `OLLAMA_STARTUP_WAIT` seconds for Ollama to become reachable before treating it as an error (so a boot-order race doesn't kill the service).

>**Caution — disk usage.** Pulled models are downloaded in full and stored in Ollama's volume (`../data/ollama`), and they are **never removed automatically**. Models are large so repeatedly changing `GENERATION_MODEL` will quietly accumulate gigabytes and can fill the host disk. Make sure the host has room for the model you configure, and periodically prune unused models with `ollama rm <model>` (or by clearing the volume).

### Setup

```bash
docker compose -f docker-compose.debug.yml up --build
```

### Config files

The following file is read from `/config` at startup (mounted via `docker-compose.debug.yml`):

| File | Purpose |
|---|---|
| `enrichment-query.rq` | SPARQL query to fetch title and content for retrieved URIs |

### Verification

```bash
curl -X POST http://localhost:8000/question-answering/answer -H "Content-Type: application/json" -d "{\"question\": \"What subsidies exist for renovating an older home?\"}"
```

```bash
curl -X POST http://localhost:8000/question-answering/answer -H "Content-Type: application/json" -d "{\"question\": \"Als ik iets aan mijn huis verbouw, ben ik dan zelf verantwoordelijk voor beschadigingen aan de inrichting van het openbaar domein, groenaanleg, bermen, trottoirs, boordstenen, straatkolken en de rijweg die te wijten zijn aan de bouwactiviteit ?\"}"
```

### Expected input

```json
{
  "question": "What subsidies exist for renovating an older home?",
  "top_n": 5,
  "localAuthority": "http://data.lblod.info/id/bestuurseenheden/6358381406fcce10a7eba9b6a1257626"
}
```

- `question`: The user question
- `top_n`: Max documents to include in the answer (default: `5`)
- `localAuthority`: Optional URI of the local authority to filter results by

### Expected output

```json
{
  "answer": "Based on the retrieved documents, ...",
  "sources": [
    {
      "uri": "https://example.org/document/1",
      "title": "Example title",
      "content": "Document content text...",
      "score": 0.812
    }
  ]
}
```

- `answer`: The generated answer from the LLM.
- `sources`: The retrieved source documents used for the answer.
  - `uri`: The document identifier returned by the retrieval API.
  - `title`: The document title resolved from the SPARQL endpoint.
  - `content`: The document content resolved from the SPARQL endpoint.
  - `score`: The similarity score from the retrieval API (may be `null`).

### Other environment variables

| Variable | Description | Default |
|---|---|---|
| `SEARCH_API_URL` | mu-search raw-DSL search endpoint (accepts a raw Elasticsearch query) | `http://search:80/expressions/large-search` |
| `SEARCH_LEXICAL_API_URL` | mu-search full-text endpoint used for the fuzzy BM25 leg | `http://search:80/expressions/search` |
| `EMBEDDING_API_URL` | Embedding service endpoint | — |
| `GENERATION_TIMEOUT` | LLM request timeout in seconds | `300.0` |
| `GENERATION_TEMPERATURE` | Sampling temperature for answer generation | `0.2` |
| `OLLAMA_PULL_TIMEOUT` | Timeout (seconds) for auto-pulling a missing Ollama model on first use | `1800.0` |
| `OLLAMA_STARTUP_WAIT` | Seconds to wait at startup for Ollama to become reachable before failing | `60.0` |
| `MAX_CONTENT_CHARS` | Max characters of document content passed to the LLM | `1000` |
| `REQUEST_TIMEOUT` | Timeout for calls to search and embedding services (seconds) | `10.0` |
| `MIN_SCORE` | Minimum dense similarity score to include a document | `0.72` |
| `EMBEDDING_NUM_CANDIDATES` | Candidate pool size (num_candidates) for kNN search | `100` |
| `HYBRID_ENABLED` | Use hybrid (dense + fuzzy BM25) retrieval; `false` falls back to dense-only | `true` |
| `RRF_K` | RRF constant for fusing the retrieval legs | `60` |
| `RRF_POOL_SIZE` | Max documents in the fused pool before reranking | `15` |
| `LEXICAL_SIZE` | Max documents requested from the fuzzy BM25 leg (before the `owning-body` post-filter) | `200` |
| `RERANK_ENABLED` | Load and use the cross-encoder reranker | `true` |
| `RERANK_MODEL_PATH` | Local path of the cross-encoder model (set by the image) | — |
| `RERANK_MAX_CHARS` | Max characters of document description fed to the cross-encoder | `512` |
| `EMBEDDING_CACHE_MAX` | Max cached question embeddings | `256` |


### Brief analysis on similarity scores

Top 50 results, `k=50`, `num_candidates=400`:

| Question | Score range |
|---|---|
| "welke smaken ijsjes zijn er?" | 0.630–0.639 |
| "als ik 2 muntjes gooi, wat is de kans dat ik 2 keer kop krijg?" | 0.674–0.699 |
| "qpwojednewd ewpirmfwef pwqoejk wef" | 0.688–0.703 |
| "kan ik bij de toeristische dienst een fiets huren?" | 0.708–0.743 |
| "wie is er verantwoordelijk voor schade aan het trottoir bij een verbouwing?" | 0.761–0.797 |
| "waar moet ik op letten als ik een halloweentocht wil organiseren?" | 0.769–0.817 |

A threshold of `0.72–0.75` filters out clearly irrelevant questions while keeping relevant ones. The current default is `0.72`.

### Storage of questions and answers

This service will store every question it receives, along with the answer it gave, including context such as the prompt and sourced that were used.
Through sparql-parser in `app-decide`, it stores these in a named graph: `http://mu.semte.ch/graphs/public/question-answering`.
To clear the question answering graph, run `CLEAR SILENT GRAPH <http://mu.semte.ch/graphs/public/question-answering>`

### LLM cost and duration tracking

Each answer also stores the generation call as an `ext:AICall` resource (linked from the answer via `ext:performedAICall`), with `ext:tokenIn`, `ext:tokenOut`, `ext:duration` (seconds) and `ext:cost` (USD). `ext:AICall` has no SHACL shape in app-decide's authorization config, so it's written straight into the `question-answering` graph via an explicit `GRAPH` clause and a `sudo` query, instead of relying on sparql-parser's shape-based routing.

Cost is priced from [OpenRouter](https://openrouter.ai/api/v1/models)'s public pricing, keyed off an `ext:openrouterId` triple on the model's URI (`urn:llm:<provider>:<model>`). No triple means no price, so cost is `0.0`. To price a cloud model, register its OpenRouter id once:

```sparql
PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
INSERT DATA {
  <urn:llm:mistralai:mistral-medium-3-5> ext:openrouterId "mistralai/mistral-medium-3-5" .
}
```
