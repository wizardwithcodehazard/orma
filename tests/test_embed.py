import numpy as np

from smruti.embed import l2_to_cosine, normalize, to_blob
from smruti.messages import message_text


def test_normalize_gives_unit_rows_and_handles_zero():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])
    assert out.dtype == np.float32


def test_l2_distance_converts_to_cosine_for_unit_vectors():
    assert l2_to_cosine(0.0) == 1.0
    assert abs(l2_to_cosine(float(np.sqrt(2.0)))) < 1e-9


def test_to_blob_is_float32_bytes():
    assert len(to_blob(np.ones(4))) == 16


def test_fake_embedder_reflects_word_overlap(embedder):
    a, b, c = embedder.embed(["user lives in pune", "user lives in mumbai", "quarterly tax filing"])
    assert a @ b > a @ c


def test_message_text_handles_strings_and_parts():
    assert message_text({"role": "user", "content": "hi"}) == "hi"
    parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}]
    assert message_text({"role": "user", "content": parts}) == "a\nb"
    assert message_text({"role": "assistant", "content": None}) == ""
