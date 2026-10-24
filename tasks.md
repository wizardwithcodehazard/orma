# Roadmap

## Done

- v1 (historical, in git): proxy and CLI, triple graph with spreading activation, priming, summaries, procedures, quotes, kartā labels, learned cardinality, timezone-aware dates, eval harness.
- v2, a general episode-first memory layer:
  - any source through one interface: `add`, `add_file`, `observe` (chat), `search`, `sleep`, `rebuild`; CLI `add`, `search`, `rebuild`;
  - records kept forever and split into indexed sentences, searchable at once;
  - sleep cites line numbers and structures only one-valued states and dated events; code checks citation, holder (pronoun carry-over), grounding and time;
  - recall: hybrid search with a bounded recency boost, current states, `[outdated]` marks;
  - eval: dev and held-out test splits, retrieval checks (`finds`), the episodic-only baseline (`enable_semantic=false`).

## Next (in this order)

1. Held-out test split once, next to dev and the episodic-only baseline.
2. LongMemEval (oracle split) with last-k, full-context and plain-RAG baselines.
3. `rebuild` with a 7B sleep model; measure the gain over 3B.
4. Grow the eval to 30–50 cases, judged by a larger model.
5. Tool-output memory for coding sessions, as a separate tested module.
6. Packaging: PyPI and `smruti up`.

## Later

- A LoRA extraction adapter trained on corrected sleep output.
- `smruti correct` and asking the user when a memory is uncertain.
- An optional MCP server.
- Token counting through the upstream `/tokenize` endpoint instead of `chars_per_token`.
