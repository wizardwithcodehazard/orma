"""
pack.py - Budgeted prompt assembly. The newest turns are always kept; memory is added in a
fixed section order (core facts first, so they form a stable, cache-friendly prefix); older
turns fill whatever budget remains.
"""

import math
from dataclasses import dataclass, field

from smruti.messages import message_text

SECTIONS = {
    "core": "Known facts about the user",
    "facts": "Relevant facts",
    "procedures": "Lessons from past actions",
    "summaries": "Background",
}


def count_tokens(text: str, chars_per_token: float) -> int:
    return math.ceil(len(text) / chars_per_token)


@dataclass
class MemoryItem:
    section: str
    id: int
    text: str


@dataclass
class Packed:
    memory_text: str
    turns: list[dict]
    used: list[MemoryItem] = field(default_factory=list)
    tokens: int = 0


def render(items: list[MemoryItem]) -> str:
    if not items:
        return ""
    lines = ["## Memory"]
    for section, title in SECTIONS.items():
        texts = [item.text for item in items if item.section == section]
        if texts:
            lines.append(f"{title}:")
            lines.extend(f"- {text}" for text in texts)
    return "\n".join(lines)


def pack(core, ranked, turns, budget: int, chars_per_token: float, min_recent: int) -> Packed:
    def cost(text: str) -> int:
        return count_tokens(text, chars_per_token)

    kept = list(turns[-min_recent:]) if min_recent > 0 else []
    older = list(turns[: len(turns) - len(kept)])
    kept_cost = sum(cost(message_text(t)) for t in kept)

    used: list[MemoryItem] = []
    for item in [*core, *ranked]:
        if cost(render(used + [item])) + kept_cost <= budget:
            used.append(item)
    memory_text = render(used)

    remaining = budget - cost(memory_text) - kept_cost
    extra: list[dict] = []
    for turn in reversed(older):
        turn_cost = cost(message_text(turn))
        if turn_cost > remaining:
            break
        extra.insert(0, turn)
        remaining -= turn_cost

    final_turns = extra + kept
    tokens = cost(memory_text) + sum(cost(message_text(t)) for t in final_turns)
    return Packed(memory_text, final_turns, used, tokens)
