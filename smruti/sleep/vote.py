"""
vote.py - Replay and vote. Sleep extracts each batch more than once (a deterministic pass, then
sampled replays) and keeps only what the passes agree on, like a memory trace that must be
reactivated before it is consolidated. Agreement is semantic: two passes may word the same
relation differently, so facts match by embedding similarity of their rendered text.
"""

from dataclasses import replace

import numpy as np

from smruti.sleep.extract import ExtractedFact, Extraction
from smruti.store.records import fact_text


def render(fact: ExtractedFact) -> str:
    return fact_text(fact.subject, fact.predicate, fact.object)


def vote(extractions: list[Extraction], embed, threshold: float) -> tuple[Extraction, list[ExtractedFact]]:
    """Splits the first extraction into what some replay confirms and the facts none does.
    A fact is agreed when a replayed fact's rendered text has cosine similarity >= threshold
    (embed returns unit vectors). Kept entities are those agreed facts reference or a replay also
    names; user names must appear in a replay too. Procedures and summaries come from the first pass."""
    first, replays = extractions[0], extractions[1:]
    if not replays:
        return first, []
    replayed = np.array([embed(render(f)) for r in replays for f in r.facts])
    agreed, unconfirmed = [], []
    for fact in first.facts:
        confirmed = len(replayed) > 0 and float(np.max(replayed @ embed(render(fact)))) >= threshold
        (agreed if confirmed else unconfirmed).append(fact)
    named = {f.subject.lower() for f in agreed} | {f.object.lower() for f in agreed}
    named |= {name.lower() for r in replays for name in r.entities}
    replayed_names = {name.lower() for r in replays for name in r.user_names}
    return replace(
        first,
        user_names=[name for name in first.user_names if name.lower() in replayed_names],
        entities={name: type_ for name, type_ in first.entities.items() if name.lower() in named},
        facts=agreed,
    ), unconfirmed
