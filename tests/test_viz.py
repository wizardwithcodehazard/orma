import pytest
from starlette.testclient import TestClient

from orma.memory import Memory
from orma.serve.proxy import create_app


def test_export_connectome(tmp_path):
    mem = Memory(path=tmp_path)
    mem.add("I live in Indiranagar, Bangalore.")
    mem.add("My dentist is Dr. Rao.")

    # Export connectome from store/memory
    data = mem.export_connectome("default")
    assert "nodes" in data
    assert "links" in data
    assert "hippocampus" in data
    assert "stats" in data

    # Verify nodes contain entity 'user'
    user_nodes = [n for n in data["nodes"] if n["type"] == "entity" and n["label"] == "user"]
    assert len(user_nodes) == 1

    # Verify hippocampus contains episodes
    assert len(data["hippocampus"]) >= 2
    mem.close()


def test_viz_proxy_endpoints(tmp_path):
    mem = Memory(path=tmp_path)
    mem.add("Alice likes green tea.", space="user_alice")

    app = create_app(mem, "http://localhost:11434/v1", default_space="user_alice")
    client = TestClient(app)

    # 1. Test connectome JSON endpoint
    res = client.get("/_orma/brain")
    assert res.status_code == 200
    brain_data = res.json()
    assert brain_data["space"] == "user_alice"
    assert "nodes" in brain_data
    assert "stats" in brain_data

    # 2. Test visualizer HTML endpoint
    res_viz = client.get("/_orma/viz")
    assert res_viz.status_code == 200
    assert "text/html" in res_viz.headers["content-type"]
    assert "3d-force-graph" in res_viz.text
    assert "NEURAL BRAIN" in res_viz.text

    # 3. Test alternate paths
    assert client.get("/brain").status_code == 200
    assert client.get("/viz").status_code == 200

    # 4. Test live recall search simulation endpoint
    res_search = client.get("/_orma/search?q=green%20tea")
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert "matched_node_ids" in search_data
    assert "prompt_block" in search_data

    # 5. Test add memory playground endpoint
    res_add = client.post("/_orma/add", json={"text": "Alice also likes matcha."})
    assert res_add.status_code == 200
    assert res_add.json()["status"] == "added"

    mem.close()

