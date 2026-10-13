from smruti.serve.capture import StreamAccumulator, cap_text, mask_secrets, new_turns, reply_text


def test_first_request_records_all_user_turns():
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
    assert new_turns(msgs) == [("user", "hi")]


def test_only_turns_after_the_last_assistant_are_new():
    msgs = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_shell", "arguments": '{"cmd": "npm run build"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Error: Cannot find module 'vite'"},
        {"role": "user", "content": "why did it fail?"},
    ]
    assert new_turns(msgs) == [
        ("tool", 'run_shell({"cmd": "npm run build"}) -> Error: Cannot find module \'vite\''),
        ("user", "why did it fail?"),
    ]


def test_unknown_tool_call_and_empty_turns():
    msgs = [{"role": "assistant", "content": "ok"}, {"role": "tool", "tool_call_id": "x", "content": "done"},
            {"role": "user", "content": "   "}]
    assert new_turns(msgs) == [("tool", "tool call -> done")]


def test_reply_text_reads_message_content():
    assert reply_text(b'{"choices": [{"message": {"role": "assistant", "content": " Hi! "}}]}') == "Hi!"
    assert reply_text(b"not json") == ""
    assert reply_text(b'{"choices": []}') == ""


def test_accumulator_handles_split_lines_and_utf8():
    acc = StreamAccumulator()
    data = ('data: {"choices":[{"delta":{"content":"नमस्ते"}}]}\n\n'
            'data: {"choices":[{"delta":{}}]}\n\n'
            'data: [DONE]\n\n').encode()
    for i in range(0, len(data), 7):
        acc.feed(data[i:i + 7])
    assert acc.text == "नमस्ते"


def test_accumulator_only_reads_choice_zero():
    acc = StreamAccumulator()
    acc.feed(b'data: {"choices":[{"index":1,"delta":{"content":"B"}},{"index":0,"delta":{"content":"A"}}]}\n\n'
             b'data: {"choices":[{"delta":{"content":"C"}}]}\n\n')
    assert acc.text == "AC"


def test_mask_secrets_by_format():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nabc\ndef\n-----END RSA PRIVATE KEY-----"
    text = (f"k=sk-proj-abcdefghijklmnop1234 Authorization: Bearer abc.def-123 AKIAABCDEFGHIJKLMNOP "
            f"ghp_{'a' * 30} {pem} https://x/y?password=hunter2&a=1 API_KEY=abc123 ok")
    out = mask_secrets(text)
    for leaked in ("sk-proj", "abc.def", "AKIA", "ghp_", "abc\ndef", "hunter2", "abc123"):
        assert leaked not in out
    assert "a=1" in out and out.endswith(" ok") and "password=[REDACTED]" in out
    assert mask_secrets("nothing secret here, sk-short") == "nothing secret here, sk-short"


def test_cap_text_keeps_head_and_tail():
    assert cap_text("short", 100) == "short"
    out = cap_text("A" * 100 + "B" * 100, 50)
    assert out.startswith("A") and out.endswith("B") and "…[truncated]…" in out and len(out) < 80
