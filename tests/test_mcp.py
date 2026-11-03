import pytest

from orma import Memory
from orma.serve.mcp import MCPServer


@pytest.fixture
def mcp_server(tmp_path):
    from orma.config import OrmaConfig
    config = OrmaConfig(data_dir=tmp_path)
    memory = Memory(config=config)
    server = MCPServer(memory=memory, default_space="test", sleep_poll_seconds=None)
    yield server
    memory.close()

@pytest.mark.anyio
async def test_mcp_initialize(mcp_server):
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"clientInfo": {"name": "test-client", "version": "1.0"}}
    }
    resp = await mcp_server.handle_request(req)
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "orma"
    assert "tools" in resp["result"]["capabilities"]

@pytest.mark.anyio
async def test_mcp_tools_list(mcp_server):
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    resp = await mcp_server.handle_request(req)
    tool_names = [t["name"] for t in resp["result"]["tools"]]
    assert "orma_recall" in tool_names
    assert "orma_remember" in tool_names
    assert "orma_facts" in tool_names
    assert "orma_forget" in tool_names

@pytest.mark.anyio
async def test_mcp_remember_and_recall_flow(mcp_server):
    # 1. Remember a decision
    rem_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "orma_remember",
            "arguments": {
                "text": "Project rule: Always use SQLite WAL mode and indices for query performance.",
                "role": "note"
            }
        }
    }
    rem_resp = await mcp_server.handle_request(rem_req)
    assert "Successfully remembered" in rem_resp["result"]["content"][0]["text"]

    # 2. Recall the decision
    rec_req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "orma_recall",
            "arguments": {
                "query": "What is the rule about database modes?"
            }
        }
    }
    rec_resp = await mcp_server.handle_request(rec_req)
    content = rec_resp["result"]["content"][0]["text"]
    assert "SQLite WAL mode" in content

@pytest.mark.anyio
async def test_mcp_ping(mcp_server):
    req = {"jsonrpc": "2.0", "id": 5, "method": "ping"}
    resp = await mcp_server.handle_request(req)
    assert resp["id"] == 5
    assert resp["result"] == {}

