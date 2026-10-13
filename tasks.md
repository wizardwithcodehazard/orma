# Roadmap

## Done

- Plan 1: memory core (store, encode, recall, sleep, SDK).
- Plan 2: proxy and CLI (`smruti proxy`, `sleep`, `facts`, `recall`, `forget`, `stats`) and documentation.
- Accuracy work from real `llama3.2:3b` runs: schema-constrained extraction, grounding filter, stated-vs-inferred time, user-name aliases with guards, tool-only procedures, optional replay voting.
- Human-memory features: episodic events, priming, ACT-R base-level activation weighted by surprise, today's date in the memory block.

## Next (in this order)

1. Deterministic dates: the model copies a relative phrase ("last month"), a date library resolves it against the turn's timestamp.
2. Show source sentences: store each fact with the sentence it came from and put the sentence (not only the triple) in the prompt, so a wrong triple cannot mislead the answer.
3. A small labelled eval set (30–50 conversations covering attribution, aliasing, relative time, missing links) run after every change.
4. Plan 3: LongMemEval harness (oracle split first) with last-k, full-context and naive-RAG baselines plus ablations of every `enable_*` switch. Runs on Kaggle's free GPU with a Gemini judge; see HANDOVER.md.
5. Rewrite-before-extract (resolve pronouns into standalone sentences, then extract) and yes/no verification of each extracted fact.
6. A fixed relation vocabulary (about 40 relations plus "other") as a schema enum; cardinality per relation becomes deterministic.
7. Optional `--sleep-model` so sleep can use a stronger model than chat.
8. Packaging: publish to PyPI and add `smruti up` (check Ollama, pull the model, start the proxy).

## Later

- GLiNER / coreference models as fast perception before the LLM.
- A LoRA extraction adapter trained on corrected sleep output (data flywheel).
- `smruti correct` and asking the user when a memory is uncertain.
- An optional MCP server.
- Token counting through the upstream `/tokenize` endpoint instead of `chars_per_token`.
- Keeping tool-call and tool-result pairs together when trimming history.
