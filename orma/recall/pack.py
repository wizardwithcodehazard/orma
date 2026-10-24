"""
pack.py - Budgeted prompt assembly. The newest turns are always kept; memory lines are chosen in priority
order (what is true now, then what was said, then events) while they fit, and shown grouped by section,
the things said in the order they were said; older turns fill whatever budget remains.
"""

from dataclasses import dataclass, field

from orma.messages import message_text
from orma.text import count_tokens

SECTIONS = {
    "states": "Now true",
    "passages": "Things said before",
    "events": "Events",
}

FRAMING = ("What you remember about the person you are talking to, from earlier conversations, notes and "
           "documents. Use it when relevant; do not mention this list. A line marked [outdated] was true when "
           "it was said and has changed since.")


@dataclass
class MemoryItem:
    section: str  # states | passages | events
    id: int
    text: str
    order: float = 0.0  # display order within its section (passages: when they were said)


@dataclass
class Packed:
    memory_text: str
    turns: list[dict]
    used: list[MemoryItem] = field(default_factory=list)
    tokens: int = 0


def framing(today: str = "") -> str:
    return f"{FRAMING} Today is {today}." if today else FRAMING


def render(items: list[MemoryItem], today: str = "") -> str:
    if not items:
        return ""
    lines = ["## Memory", framing(today)]
    for section, title in SECTIONS.items():
        chosen = sorted((i for i in items if i.section == section), key=lambda i: (i.order, i.id))
        if chosen:
            lines.append(f"{title}:")
            lines.extend(f"- {item.text}" for item in chosen)
    return "\n".join(lines)


def pack(candidates: list[MemoryItem], turns: list[dict], budget: int, chars_per_token: float, min_recent: int,
         today: str = "", state_cap: int | None = None) -> Packed:
    """candidates in priority order; states beyond state_cap tokens are left out."""
    def cost(text: str) -> int:
        return count_tokens(text, chars_per_token)

    kept = list(turns[-min_recent:]) if min_recent > 0 else []
    older = list(turns[: len(turns) - len(kept)])
    kept_cost = sum(cost(message_text(t)) for t in kept)

    used: list[MemoryItem] = []
    state_tokens = 0
    for item in candidates:
        line_cost = cost(f"- {item.text}\n")
        if item.section == "states" and state_cap is not None and state_tokens + line_cost > state_cap:
            continue
        if cost(render(used + [item], today)) + kept_cost <= budget:
            used.append(item)
            state_tokens += line_cost if item.section == "states" else 0
    memory_text = render(used, today)

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
