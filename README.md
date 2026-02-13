## UC2 Subsidies RAG Stub

This stub shows the HTTP flow and defines the input/output formats for a simple subsidies RAG (Retrieval-Augmented Generation) system.

The flow is based on the following:
- Create an HTTP service that accepts a question (or dialog) in JSON format.
- Take the question and perform a semantic search.
- Pass the question plus the retrieved decisions (top N, filtered by a relevance threshold) to an LLM to generate a response.
- Return the generated response together with the URIs of the retrieved decisions.

### Setup

```bash
docker-compose up --build
```

### Verification

```bash
curl -X POST http://localhost:8000/uc2/answer -H "Content-Type: application/json" -d "{\"question\": \"What subsidies exist for renovating an older home?\"}"
```

Example with dialog context:
```bash
curl -X POST http://localhost:8000/uc2/answer -H "Content-Type: application/json" -d "{\"question\": \"What subsidies exist for renovating an older home?\", \"dialog\": [{\"role\": \"user\", \"content\": \"Hello, I need help with home renovation subsidies.\"}, {\"role\": \"assistant\", \"content\": \"Hello! I'd be happy to help you with home renovation subsidies. What would you like to know?\"}]}"
```

**Format notes:**
- `question`: The current user question being asked
- `dialog`: Previous conversation history (optional). Should contain alternating `user`/`assistant` messages and end with an `assistant` message if present
- The current `question` is separate from `dialog` and represents the new user message
- *Note: Future changes could include making the current question part of the dialog array for a more chatbot-like experience*

The current implementation is a pure stub: semantic search, thresholding, and LLM answer generation all return mock data to demonstrate the end-to-end flow.
