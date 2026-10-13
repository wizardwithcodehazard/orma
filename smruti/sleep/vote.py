"""
vote.py - Replay and vote. Sleep may extract each batch more than once (a deterministic pass,
then sampled replays). The first pass is what gets consolidated; a fact a replay also produced
is agreed and earns extra confidence, like a memory trace reactivated before it is consolidated.
Agreement is semantic: two passes may word the same relation differently, so facts match by
embedding similarity of their rendered text.
"""

from dataclasses import replace

import numpy as np

from smruti.sleep.extract import ExtractedFact, Extraction
from smruti.store.records import fact_text


def render(fact: ExtractedFact) -> str:
    return fact_text(fact.subject, fact.predicate, fact.object)


def vote(extractions: list[Extraction], embed, threshold: float) -> tuple[Extraction, set[int]]:
    """The first extraction, with user names from every pass, and the indices of its facts that some
    replay confirms: a replayed fact's rendered text has cosine similarity >= threshold (embed returns
    unit vectors). Entities, events, procedures and summaries come from the first pass."""
    first, replays = extractions[0], extractions[1:]
    replayed = np.array([embed(render(f)) for r in replays for f in r.facts])
    agreed = {i for i, fact in enumerate(first.facts)
              if len(replayed) and float(np.max(replayed @ embed(render(fact)))) >= threshold}
    names: dict[str, str] = {}
    for extraction in extractions:
        for name in extraction.user_names:
            names.setdefault(name.lower(), name)
    return replace(first, user_names=list(names.values())), agreed
