import functools

"""
mcp.py - Native Model Context Protocol (MCP) server for Orma.
Allows coding assistants and agents (Claude Code, Cursor, Windsurf, Roo Code, Hermes, etc.)
to recall past architectural decisions, persist context across sessions, and manage facts
via standard JSON-RPC 2.0 over stdio without requiring heavy dependencies.
"""

import asyncio
import json
import logging
import sys
from typing import Any

import anyio

from orma import __version__
from orma.memory import Memory
from orma.serve.scheduler import SleepScheduler

logger = logging.getLogger(__name__)

# Protocol version supported
PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "orma_recall",
        "description": "Recall long-term memories, architectural decisions, past user preferences, and constraints under a token budget with Sutra dense packing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (e.g. 'project architecture', 'database configuration', 'user preferences')."
                },
                "space": {
                    "type": "string",
                    "description": "Optional isolated memory space (defaults to 'default')."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "orma_remember",
        "description": "Store an architectural decision, requirement, or inhibitory anti-pattern/forbidden mistake permanently into long-term memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The fact, requirement, decision, or note to record."
                },
                "kind": {
                    "type": "string",
                    "enum": ["fact", "anti_pattern"],
                    "default": "fact",
                    "description": "Set to 'anti_pattern' if this is a mistake, prohibited syntax, or negative rule to avoid."
                },
                "role": {
                    "type": "string",
                    "enum": ["note", "user", "assistant", "document"],
                    "default": "note",
                    "description": "The provenance role of this memory."
                },
                "space": {
                    "type": "string",
                    "description": "Optional isolated memory space (defaults to 'default')."
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "orma_forbid",
        "description": "Directly register an anti-pattern, prohibited syntax, or mistake to NEVER repeat.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule": {
                    "type": "string",
                    "description": "The forbidden mistake or action to never take (e.g. 'Never use semicolon in Windows terminal commands')."
                },
                "space": {
                    "type": "string",
                    "description": "Optional isolated memory space (defaults to 'default')."
                }
            },
            "required": ["rule"]
        }
    },
    {
        "name": "orma_facts",
        "description": "Inspect all currently active canonical truths, project rules, and known facts without performing a search query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "space": {
                    "type": "string",
                    "description": "Optional isolated memory space (defaults to 'default')."
                },
                "include_history": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to also include closed / outdated past facts."
                }
            }
        }
    },
    {
        "name": "orma_forget",
        "description": "Mark an old obsolete fact or superseded constraint as no longer active.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state_id": {
                    "type": "integer",
                    "description": "The ID of the state to forget/close."
                },
                "space": {
                    "type": "string",
                    "description": "Optional isolated memory space (defaults to 'default')."
                }
            },
            "required": ["state_id"]
        }
    }
]


class MCPServer:
    def __init__(self, memory: Memory, default_space: str = "default", sleep_poll_seconds: float = 60.0):
        self.memory = memory
        self.default_space = default_space
        self.scheduler = SleepScheduler(memory, sleep_poll_seconds) if sleep_poll_seconds else None

    async def run(self):
        """Runs the MCP server reading JSON-RPC lines from stdin and writing to stdout."""
        if self.scheduler is not None:
            self.scheduler.start()

        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                raw = line.decode("utf-8").strip()
                if not raw:
                    continue
                try:
                    request = json.loads(raw)
                    response = await self.handle_request(request)
                    if response is not None:
                        out = json.dumps(response) + "\n"
                        sys.stdout.write(out)
                        sys.stdout.flush()
                except Exception as e:
                    logger.exception("Error handling MCP request: %s", raw)
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32603, "message": f"Internal error: {e}"}
                    }
                    sys.stdout.write(json.dumps(err_resp) + "\n")
                    sys.stdout.flush()
        finally:
            if self.scheduler is not None:
                try:
                    await anyio.to_thread.run_sync(self.scheduler.flush)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("MCP shutdown scheduler flush failed: %s", exc)
                self.scheduler.stop()

    async def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        method = req.get("method")
        msg_id = req.get("id")

        # Notifications (no msg_id) do not receive a response
        if method == "notifications/initialized":
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False}
                    },
                    "serverInfo": {
                        "name": "orma",
                        "version": __version__
                    }
                }
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": TOOLS
                }
            }

        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            result = await self.execute_tool(name, args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result
            }

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {}
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

    async def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        space = args.get("space") or self.default_space
        try:
            if name == "orma_recall":
                query = args.get("query", "")
                result = await anyio.to_thread.run_sync(
                    functools.partial(self.memory.build_context, [{"role": "user", "content": query}], space=space)
                )
                system_prompt = next((m["content"] for m in result if m.get("role") == "system"), "")
                return {
                    "content": [{"type": "text", "text": system_prompt or "No relevant memories found."}]
                }

            elif name == "orma_remember":
                text = args.get("text", "")
                kind = args.get("kind", "fact")
                role = args.get("role", "note")
                record_text = f"FORBIDDEN RULE: {text}" if kind == "anti_pattern" else text
                results = await anyio.to_thread.run_sync(self.memory.add, record_text, space, role)
                ep_ids = [str(r.episode_id) for r in results]
                total_sentences = sum(r.sentences for r in results)
                tag = " (anti-pattern guardrail)" if kind == "anti_pattern" else ""
                return {
                    "content": [{"type": "text", "text": f"Successfully remembered{tag}: {text} (episodes: {', '.join(ep_ids)}, sentences: {total_sentences})"}]
                }

            elif name == "orma_forbid":
                rule = args.get("rule", "")
                record_text = f"FORBIDDEN RULE: {rule}"
                results = await anyio.to_thread.run_sync(self.memory.add, record_text, space, "note")
                return {
                    "content": [{"type": "text", "text": f"Successfully registered inhibitory constraint: [FORBIDDEN: {rule}]"}]
                }

            elif name == "orma_facts":
                include_history = bool(args.get("include_history", False))
                facts = await anyio.to_thread.run_sync(
                    functools.partial(self.memory.facts, space=space, history=include_history)
                )
                lines = [f"Active facts for space '{space}':"]
                for f in facts:
                    status = f" [outdated: replaced by {f['superseded_by']}]" if f.get("superseded_by") else ""
                    lines.append(f"- ID {f['id']}: {f['text']}{status}")
                if len(lines) == 1:
                    lines.append("(No facts recorded yet)")
                return {
                    "content": [{"type": "text", "text": "\n".join(lines)}]
                }

            elif name == "orma_forget":
                state_id = int(args.get("state_id"))
                ok = await anyio.to_thread.run_sync(
                    functools.partial(self.memory.forget, state_id, space=space)
                )
                return {
                    "content": [{"type": "text", "text": f"Fact {state_id} ended: {ok}"}]
                }

            else:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Unknown tool: {name}"}]
                }
        except Exception as e:
            logger.exception("Error executing tool %s", name)
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool execution failed: {e!s}"}]
            }


def run_mcp_server(memory: Memory, default_space: str = "default"):
    """Entry point for running the MCP server synchronously."""
    server = MCPServer(memory, default_space=default_space)
    asyncio.run(server.run())
