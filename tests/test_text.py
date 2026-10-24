from orma.text import count_tokens, made_of, phrases, span_of, split_sentences, words


def test_sentences_split_at_ends_and_lines():
    text = "Hi! I'm Sahil.  I live in Pune?\n\n- bought milk\n- paid rent"
    assert split_sentences(text) == ["Hi!", "I'm Sahil.", "I live in Pune?", "- bought milk", "- paid rent"]


def test_initials_and_short_abbreviations_do_not_end_a_sentence():
    assert split_sentences("My dentist is Dr. Rao in Kothrud. He is good.") == [
        "My dentist is Dr. Rao in Kothrud.", "He is good."]
    assert split_sentences("Bring fruit, e.g. mangoes. Thanks.") == ["Bring fruit, e.g. mangoes.", "Thanks."]
    assert split_sentences("Mail J. K. Rowling today.") == ["Mail J. K. Rowling today."]
    assert split_sentences("I am Sam. I live here.") == ["I am Sam.", "I live here."]


def test_code_blocks_become_one_marker():
    text = "Here is the fix:\n```python\nimport os\nprint(os.getcwd())\n```\nRun it again."
    assert split_sentences(text) == ["Here is the fix:", "[code block: python, 2 lines]", "Run it again."]


def test_long_sentences_are_cut_at_words_and_empty_pieces_dropped():
    pieces = split_sentences("word " * 50, max_chars=40)
    assert all(len(p) <= 40 for p in pieces) and " ".join(pieces) == ("word " * 50).strip()
    assert split_sentences("... --- !!!") == []


def test_word_helpers():
    assert words("Is_allergic to, Peanuts!") == ["is", "allergic", "to", "peanuts"]
    assert span_of("my sister Asha Rao", "Asha Rao") == (2, 4) and span_of("Ashaa", "Asha") is None
    assert made_of("She", {"she"}) and not made_of("my sister", {"my"}) and not made_of("", {"she"})
    assert phrases("my mom, my mom") == ["my", "mom", "my mom", "mom my"]
    assert count_tokens("abcdefg", 3.5) == 2
