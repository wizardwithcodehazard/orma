import pytest
from orma.sleep.extract import parse_extraction, ExtractedState
from orma.recall.pack import MemoryItem, is_inhibitory, format_sutra, render
from orma.config import OrmaConfig
from orma import Memory
from orma.serve.mcp import MCPServer


def test_parse_extraction_prohibited():
    data = {
        "user_name": "Vedant",
        "states": [
            {
                "s": 1,
                "about": "user",
                "attribute": "powershell_chaining",
                "value": "never use semicolon to chain commands",
                "prohibited": True,
            },
            {
                "s": 2,
                "about": "user",
                "attribute": "editor",
                "value": "neovim",
            }
        ]
    }
    extracted = parse_extraction(data)
    assert len(extracted.states) == 2
    assert extracted.states[0].prohibited is True
    assert extracted.states[0].attribute == "powershell_chaining"
    assert extracted.states[1].prohibited is False


def test_consolidator_prefixes_forbidden(tmp_path):
    config = OrmaConfig(data_dir=tmp_path)
    memory = Memory(config=config)
    try:
        from orma.sleep.consolidate import Consolidator, ConsolidationReport

        store = memory.store
        consolidator = Consolidator(store, config)
        space = "test"

        user_id = store.add_entity(space, "user", "person")
        enc_res = memory.add("Never use semicolon in powershell", space=space, role="user")
        s1 = store.episode_sentences([enc_res[0].episode_id])[0]

        item1 = ExtractedState(s=s1.id, about="user", attribute="powershell_chain", value="never use semicolon", prohibited=True)
        report = ConsolidationReport()
        consolidator._state(space, user_id, item1, s1, report)

        # Attribute in store should now be prefixed with forbidden_
        attr = store.find_attribute(space, "forbidden_powershell_chain")
        assert attr is not None
        assert attr.name == "forbidden_powershell_chain"

        states = store.current_states(space, user_id, attr.id)
        assert len(states) == 1
        assert states[0].attribute == "forbidden_powershell_chain"
        assert "forbidden powershell chain" in states[0].text

        # Test idempotency: if already has forbidden_, don't double prefix
        item2 = ExtractedState(s=s1.id, about="user", attribute="forbidden_powershell_chain", value="never use semicolon", prohibited=True)
        consolidator._state(space, user_id, item2, s1, report)
        double_attr = store.find_attribute(space, "forbidden_forbidden_powershell_chain")
        assert double_attr is None
    finally:
        memory.close()


def test_inhibitory_pack_and_sutra_hoisting():
    item_forbid = MemoryItem("states", 1, "user · forbidden_terminal_syntax: never chain commands with semicolon")
    item_forbid_space = MemoryItem("states", 4, "user · forbidden terminal syntax: never chain commands with semicolon")
    item_normal = MemoryItem("states", 2, "user · editor: neovim")
    item_log = MemoryItem("passages", 3, "2026-10-25 user: push code to origin main")

    assert is_inhibitory(item_forbid) is True
    assert is_inhibitory(item_forbid_space) is True
    assert is_inhibitory(item_normal) is False

    # Sutra formatting
    sutra_forbid = format_sutra(item_forbid)
    assert sutra_forbid == "[FORBIDDEN: never chain commands with semicolon]"
    sutra_forbid_space = format_sutra(item_forbid_space)
    assert sutra_forbid_space == "[FORBIDDEN: never chain commands with semicolon]"
    sutra_normal = format_sutra(item_normal)
    assert sutra_normal == "[user.editor: neovim]"

    # Sutra rendering: forbidden items are hoisted to the top above normal states
    rendered_sutra = render([item_normal, item_forbid, item_log], sutra=True)
    lines = rendered_sutra.strip().split("\n")
    # First line after header/framing should be the hoisted forbidden item
    assert "[FORBIDDEN: never chain commands with semicolon]" in lines[2]
    # Normal state should appear after forbidden item
    forbid_idx = lines.index("[FORBIDDEN: never chain commands with semicolon]")
    normal_idx = lines.index("[user.editor: neovim]")
    assert forbid_idx < normal_idx

    # Non-sutra rendering: forbidden items appear under Forbidden / Anti-patterns header
    rendered_standard = render([item_normal, item_forbid], sutra=False)
    assert "Forbidden / Anti-patterns:" in rendered_standard
    assert "- user · forbidden_terminal_syntax: never chain commands with semicolon" in rendered_standard
    assert "Now true:" in rendered_standard


@pytest.mark.anyio
async def test_mcp_forbid_and_anti_pattern(tmp_path):
    config = OrmaConfig(data_dir=tmp_path)
    memory = Memory(config=config)
    server = MCPServer(memory=memory, default_space="test", sleep_poll_seconds=None)
    try:
        # 1. Test tools/list includes orma_forbid
        list_resp = await server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = [t["name"] for t in list_resp["result"]["tools"]]
        assert "orma_forbid" in names
        assert "orma_remember" in names

        # 2. Test orma_forbid tool call
        forbid_resp = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "orma_forbid",
                "arguments": {
                    "rule": "Never chain commands using semicolons on Windows PowerShell; use separate lines or &&."
                }
            }
        })
        content = forbid_resp["result"]["content"][0]["text"]
        assert "Successfully registered inhibitory constraint" in content
        assert "[FORBIDDEN: Never chain commands using semicolons" in content

        # 3. Test orma_remember with kind='anti_pattern'
        rem_resp = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "orma_remember",
                "arguments": {
                    "text": "Never run rm -rf on production server",
                    "kind": "anti_pattern"
                }
            }
        })
        rem_content = rem_resp["result"]["content"][0]["text"]
        assert "Successfully remembered (anti-pattern guardrail)" in rem_content

        # 4. Recall check
        recall_resp = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "orma_recall",
                "arguments": {
                    "query": "How should I chain commands in powershell?"
                }
            }
        })
        recall_text = recall_resp["result"]["content"][0]["text"]
        assert "semicolon" in recall_text.lower()
    finally:
        memory.close()
