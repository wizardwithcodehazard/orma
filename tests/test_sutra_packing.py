from orma.recall.pack import MemoryItem, format_sutra, pack


def test_sutra_format_states_events_passages():
    st = MemoryItem("states", 1, "user · home city: Bangalore (since 2026-10-17)")
    ev = MemoryItem("events", 2, "2026-10-12: user went trekking at Sinhagad with Asha")
    ps = MemoryItem("passages", 3, "2026-10-10 user: I'm Sahil and I live in Pune. [outdated: now Bangalore]")

    assert format_sutra(st) == "[user.home city: Bangalore (since 2026-10-17)]"
    assert format_sutra(ev) == "[EVENT: 2026-10-12: user went trekking at Sinhagad with Asha]"
    assert format_sutra(ps) == "[LOG: 2026-10-10 user: I'm Sahil and I live in Pune. [outdated: now Bangalore]]"

def test_sutra_token_compression_ratio():
    candidates = [
        MemoryItem("states", 1, "user · favorite morning drink: black coffee"),
        MemoryItem("states", 2, "user · morning drink during sore throat: warm ginger tea with honey"),
        MemoryItem("states", 3, "user · current location: Bangalore (since 2026-10-17)"),
        MemoryItem("events", 4, "2026-10-12: user went trekking at Sinhagad with Asha"),
    ]
    turns = [{"role": "user", "content": "What should I drink?"}]

    # Render standard prose vs Sutra
    standard_packed = pack(candidates, turns, budget=2000, chars_per_token=3.5, min_recent=1, sutra=False)
    sutra_packed = pack(candidates, turns, budget=2000, chars_per_token=3.5, min_recent=1, sutra=True)

    # Sutra eliminates verbose section titles ("Now true:", "Events:", bullet dashes)
    assert len(sutra_packed.memory_text) < len(standard_packed.memory_text)
    assert "[user.favorite morning drink: black coffee]" in sutra_packed.memory_text
    assert "[EVENT: 2026-10-12: user went trekking at Sinhagad with Asha]" in sutra_packed.memory_text

def test_conditional_extraction_and_defeasible_boundary():
    from orma.sleep.consolidate import has_conditional_framing
    from orma.sleep.extract import ExtractedState

    # Schema-level explicit condition attribute determines defeasible boundary
    st_conditional = ExtractedState(s=1, about="user", attribute="drink", value="ginger tea", condition="sore throat")
    assert has_conditional_framing(condition=st_conditional.condition)

    # Categorical facts have empty condition (not defeasible)
    st_categorical = ExtractedState(s=2, about="user", attribute="home_city", value="Bangalore", condition="")
    assert not has_conditional_framing(condition=st_categorical.condition)

    # Pure schema-driven: arbitrary text never triggers false condition without schema attribute
    assert not has_conditional_framing(condition="", text="I bought a gift with a sharp knife on the night shift")
    assert not has_conditional_framing(condition="", text="When Harry Met Sally is my favorite movie")

