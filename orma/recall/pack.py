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
           "documents. Answer directly using these facts and statements without asking the user to re-confirm details "
           "already provided here; do not mention this list. A line marked [outdated] was true when "
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


def is_inhibitory(item: MemoryItem) -> bool:
    """Checks if a memory item represents a negative constraint / anti-pattern."""
    t = item.text.lower()
    if item.section != "states":
        return False
    if any(m in t for m in (" · forbidden_", " · forbidden ", " · avoid_", " · avoid ", " · anti_pattern", " · anti pattern", " · negative_", " · negative ")):
        return True
    if ":" in t:
        _, val = t.split(":", 1)
        val = val.strip()
        if val.startswith(("never ", "do not ", "don't ", "must not ", "avoid ", "cannot ")):
            return True
    return False


def format_sutra(item: MemoryItem) -> str:
    """Formats a memory item in dense, unambiguous Sutra format [TAG: ...]."""
    text = item.text.strip()
    if item.section == "states":
        if is_inhibitory(item):
            # Format: user · forbidden_terminal_syntax: do not use semicolon -> [FORBIDDEN: do not use semicolon]
            if " · " in text:
                _, rest = text.split(" · ", 1)
                val = rest.split(":", 1)[1].strip() if ":" in rest else rest
                return f"[FORBIDDEN: {val}]"
            return f"[FORBIDDEN: {text}]"
        if " · " in text:
            subj, rest = text.split(" · ", 1)
            return f"[{subj}.{rest}]"
        return f"[STATE: {text}]"
    elif item.section == "events":
        return f"[EVENT: {text}]"
    else:  # passages
        return f"[LOG: {text}]"


def render(items: list[MemoryItem], today: str = "", sutra: bool = False) -> str:
    if not items:
        return ""
    inhibitory = [i for i in items if is_inhibitory(i)]
    frame = framing(today)
    if inhibitory:
        frame += " Lines marked [FORBIDDEN: ...] or under Forbidden are strict negative constraints: you must NEVER suggest, mention, or take what is forbidden."
    if sutra:
        lines = ["## Memory", frame]
        # Hoist inhibitory / FORBIDDEN constraints to the very top (highest attention saliency)
        inhibitory = [i for i in items if is_inhibitory(i)]
        if inhibitory:
            lines.extend(format_sutra(i) for i in sorted(inhibitory, key=lambda i: (i.order, i.id)))

        for section in ("states", "events", "passages"):
            chosen = sorted((i for i in items if i.section == section and not is_inhibitory(i)),
                            key=lambda i: (i.order, i.id))
            lines.extend(format_sutra(item) for item in chosen)
        return "\n".join(lines)

    lines = ["## Memory", frame]
    if inhibitory:
        lines.append("Forbidden / Anti-patterns:")
        lines.extend(f"- {item.text}" for item in sorted(inhibitory, key=lambda i: (i.order, i.id)))

    for section, title in SECTIONS.items():
        chosen = sorted((i for i in items if i.section == section and not is_inhibitory(i)), key=lambda i: (i.order, i.id))
        if chosen:
            lines.append(f"{title}:")
            lines.extend(f"- {item.text}" for item in chosen)
    return "\n".join(lines)


def pack(candidates: list[MemoryItem], turns: list[dict], budget: int, chars_per_token: float, min_recent: int,
         today: str = "", state_cap: int | None = None, sutra: bool = False) -> Packed:
    """candidates in priority order; states beyond state_cap tokens are left out."""
    def cost(text: str) -> int:
        return count_tokens(text, chars_per_token)

    kept = list(turns[-min_recent:]) if min_recent > 0 else []
    older = list(turns[: len(turns) - len(kept)])
    kept_cost = sum(cost(message_text(t)) for t in kept)

    used: list[MemoryItem] = []
    state_tokens = 0
    for item in candidates:
        line_repr = format_sutra(item) + "\n" if sutra else f"- {item.text}\n"
        line_cost = cost(line_repr)
        if item.section == "states" and state_cap is not None and state_tokens + line_cost > state_cap:
            continue
        if cost(render(used + [item], today, sutra=sutra)) + kept_cost <= budget:
            used.append(item)
            state_tokens += line_cost if item.section == "states" else 0
    memory_text = render(used, today, sutra=sutra)

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