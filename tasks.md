# Roadmap

## Done

- Plan 1: memory core (store, encode, recall, sleep, SDK).
- Plan 2: proxy and CLI (`smruti proxy`, `sleep`, `facts`, `recall`, `forget`, `stats`) and documentation.

## Next

- Plan 3: LongMemEval harness with last-k, full-context and naive-RAG baselines, plus ablations of the `enable_*` switches. End-to-end runs with a real local model.

## Later

- An optional MCP server.
- Token counting through the upstream `/tokenize` endpoint instead of the `chars_per_token` estimate.
- A relevance cutoff for weak lexical matches.
- Keeping tool-call and tool-result pairs together when trimming history.
- Summary topic collision handling.
