## decide-subsidy-widget

This stub shows the HTTP flow and defines the input/output formats for a simple subsidies RAG (Retrieval-Augmented Generation) system.

The flow is based on the following:
- Create an HTTP service that accepts a question (or dialog) in JSON format.
- Take the question and perform a semantic search (optionally filtered to subsidies).
- Pass the question plus the retrieved decisions (top N, filtered by a relevance threshold) to an LLM to generate a response.
- Return the generated response together with the URIs of the retrieved decisions.

### Setup
- **Install dependencies**: `pip install -r requirements.txt`
- **Start the stub server**: `python subsidies_stub.py --server`

### Usage
- **Endpoint**: `POST /uc2/answer`
- **Content-Type**: `application/json`

### Expected Input
```json
{
  "question": "What subsidies exist for renovating an older home?",
  "dialog": [
    { "role": "user", "content": "Previous question or context" }
  ],
  "filters": { "subsidiesOnly": true },
  "top_n": 5,
  "min_score": 0.35
}
```

Only `question` is required; all other fields are optional.

### Expected Output
```json
{
  "answer": "STUB: Based on N retrieved decisions, here is a placeholder answer...",
  "sources": [
    {
      "uri": "http://example.org/decisions/123",
      "score": 0.82,
      "title": "Decision on renovation subsidy eligibility"
    }
  ]
}
```

The current implementation is a pure stub: semantic search, thresholding, and LLM answer generation all return mock data to demonstrate the end-to-end flow.