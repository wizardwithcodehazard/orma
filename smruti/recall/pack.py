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
    "episodes": "Recent notes",
}


FRAMING = ("Things remembered from earlier conversations with this user (the person you are talking to). "
           "Use them when relevant; do not mention this list.")


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
    lines = ["## Memory", FRAMING]
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

    # Exact running size of render(used): items arrive in section order, so each new section
    # adds a title line and every line after the first adds a newline.
    used: list[MemoryItem] = []
    chars = lines = 0
    titled: set[str] = set()
    for item in [*core, *ranked]:
        added = [f"- {item.text}"]
        if item.section not in titled:
            added.insert(0, f"{SECTIONS[item.section]}:")
        if lines == 0:
            added[:0] = ["## Memory", FRAMING]
        with_item = chars + sum(len(line) for line in added) + len(added) - (lines == 0)
        if math.ceil(with_item / chars_per_token) + kept_cost <= budget:
            used.append(item)
            titled.add(item.section)
            chars, lines = with_item, lines + len(added)
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
