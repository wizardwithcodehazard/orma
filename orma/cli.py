"""cli.py - The `orma` command: add to memory, search it, consolidate it, inspect it, and run the chat proxy."""

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path

import typer

from orma import __version__, clock
from orma.memory import Memory

DEFAULT_UPSTREAM = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint

app = typer.Typer(
    help="orma: local episode-first long-term memory for small LLMs.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool):
    if value:
        typer.echo(f"orma {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
):
    pass


def _memory(upstream: str | None = None, model: str | None = None) -> Memory:
    return Memory(upstream=upstream, model=model)


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def _get_active_proxy_url(host: str = "127.0.0.1", port: int = 8080) -> str | None:
    """Checks if an Orma proxy is actively serving requests on host:port."""
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/_orma/brain",
            headers={"User-Agent": "orma-cli"},
        )
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            if resp.status == 200:
                return f"http://{host}:{port}"
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    return None


@app.command()
def add(
    content: str = typer.Argument(..., help="A file path, or the text itself."),
    space: str = typer.Option("default", help="Memory space."),
    role: str | None = typer.Option(None, help="note (your own words) or document (default: document for a "
                                               "file, note for text)."),
):
    """Add a note or a text file to memory; it is searchable at once and consolidated at the next sleep."""
    if not content or not content.strip():
        typer.echo("Error: content cannot be empty.")
        raise typer.Exit(1)

    memory = _memory()
    try:
        path = Path(content)
        if path.is_file():
            results = memory.add_file(path, space, role=role or "document")
        else:
            results = memory.add(content, space, role=role or "note")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Add failed: {exc}")
        raise typer.Exit(1)
    finally:
        memory.close()
    kept = [r for r in results if r.stored]
    typer.echo(f"Kept {len(kept)} of {len(results)} records ({sum(r.sentences for r in kept)} sentences).")


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for."),
    space: str = typer.Option("default", help="Memory space."),
):
    """Show what memory holds for QUERY: states true now, sentences, events."""
    if not query or not query.strip():
        typer.echo("Error: query cannot be empty.")
        raise typer.Exit(1)

    memory = _memory()
    try:
        found = memory.search(query, space)
        tz = clock.zone(memory.config.timezone)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Search failed: {exc}")
        raise typer.Exit(1)
    finally:
        memory.close()
    if found.states:
        typer.echo("Now true:")
        for state in found.states:
            typer.echo(f"  {state.text}")
    if found.passages:
        typer.echo("Things said:")
        for sentence, score in found.passages:
            note = f"  [outdated: now {found.outdated[sentence.id].text}]" if sentence.id in found.outdated else ""
            typer.echo(f"  {score:.3f}  {sentence.label(tz)}{note}")
    if found.events:
        typer.echo("Events:")
        for event, score in found.events:
            typer.echo(f"  {score:.3f}  {event.label(tz)}")


@app.command()
def proxy(
    upstream: str = typer.Option(DEFAULT_UPSTREAM, help="OpenAI-compatible base URL of your local runtime."),
    host: str = typer.Option("127.0.0.1", help="Interface to listen on."),
    port: int = typer.Option(8080, help="Port to listen on."),
    model: str | None = typer.Option(None, help="Model for sleep consolidation (default: the model clients request)."),
    sleep_upstream: str | None = typer.Option(
        None, help="Base URL of the runtime for sleep consolidation (default: --upstream), e.g. a second server "
                   "with a stronger model."),
    space: str = typer.Option("default", help="Memory space for requests that name none."),
    sleep_poll: float = typer.Option(30.0, help="Seconds between background sleep checks; 0 disables."),
    open_browser: bool = typer.Option(False, "--open", "-o", help="Open the Web UI in your browser automatically."),
):
    """Run the memory proxy; point any OpenAI-compatible client at http://HOST:PORT/v1."""
    import threading
    import webbrowser

    import uvicorn

    from orma.serve.proxy import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    target_port = port
    if _is_port_in_use(host, target_port):
        orig_port = target_port
        while _is_port_in_use(host, target_port):
            target_port += 1
        typer.echo(f"Port {orig_port} is busy. Binding proxy to port {target_port} instead.")

    memory = _memory(sleep_upstream or upstream, model)
    sleeps_on = f", sleep -> {sleep_upstream}" if sleep_upstream else ""
    ui_url = f"http://{host}:{target_port}/#/"
    typer.echo(f"orma proxy on http://{host}:{target_port}/v1 -> {upstream}{sleeps_on} (memory: {memory.config.db_path})")
    typer.echo(f"Web UI available at: {ui_url}")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(ui_url)).start()

    app_ = create_app(memory, upstream, default_space=space, sleep_poll_seconds=sleep_poll or None)
    try:
        uvicorn.run(app_, host=host, port=target_port, log_level="warning")
    except (KeyboardInterrupt, SystemExit):
        typer.echo("Orma proxy stopped.")
    finally:
        memory.close()


@app.command()
def up(
    upstream: str = typer.Option(DEFAULT_UPSTREAM, help="OpenAI-compatible base URL of your local runtime."),
    host: str = typer.Option("127.0.0.1", help="Interface to listen on."),
    port: int = typer.Option(8080, help="Port to listen on."),
    model: str | None = typer.Option(None, help="Model for sleep consolidation."),
    sleep_upstream: str | None = typer.Option(None, help="Base URL for sleep consolidation."),
    space: str = typer.Option("default", help="Memory space."),
    sleep_poll: float = typer.Option(30.0, help="Seconds between background sleep checks; 0 disables."),
):
    """Start the Orma proxy and open the Web UI in your browser immediately."""
    proxy(
        upstream=upstream,
        host=host,
        port=port,
        model=model,
        sleep_upstream=sleep_upstream,
        space=space,
        sleep_poll=sleep_poll,
        open_browser=True,
    )


@app.command()
def viz(
    upstream: str = typer.Option(DEFAULT_UPSTREAM, help="OpenAI-compatible base URL of your local runtime."),
    host: str = typer.Option("127.0.0.1", help="Interface to listen on."),
    port: int = typer.Option(8080, help="Port to listen on."),
    model: str | None = typer.Option(None, help="Model for sleep consolidation."),
    sleep_upstream: str | None = typer.Option(None, help="Base URL for sleep consolidation."),
    space: str = typer.Option("default", help="Memory space to visualize."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the 3D Connectome in your browser."),
):
    """Launch the interactive 3D Cognitive Connectome visualizer."""
    import threading
    import webbrowser

    import uvicorn

    from orma.serve.proxy import create_app

    target_port = port

    # If the requested port is in use, check if Orma is already running on it
    if _is_port_in_use(host, target_port):
        active_url = _get_active_proxy_url(host, target_port)
        if active_url:
            ui_url = f"{active_url}/_orma/viz"
            typer.echo(f"Orma proxy is already active on port {target_port}.")
            typer.echo(f"Opening 3D Connectome at: {ui_url}")
            if open_browser:
                webbrowser.open(ui_url)
            return

        orig_port = target_port
        while _is_port_in_use(host, target_port):
            target_port += 1
        typer.echo(f"Port {orig_port} is busy. Using port {target_port} instead.")

    memory = _memory(sleep_upstream or upstream, model)
    ui_url = f"http://{host}:{target_port}/_orma/viz"
    typer.echo(f"Orma 3D Connectome running at: {ui_url}")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(ui_url)).start()

    app_ = create_app(memory, upstream, default_space=space, sleep_poll_seconds=None)
    try:
        uvicorn.run(app_, host=host, port=target_port, log_level="warning")
    except (KeyboardInterrupt, SystemExit):
        typer.echo("Orma visualizer stopped.")
    finally:
        memory.close()


def _report(report) -> str:
    return (f"Consolidated {report.episodes} records in {report.batches} batches "
            f"({report.failed_batches} failed): {report.consolidation}")


@app.command()
def sleep(
    model: str | None = typer.Option(None, help="Model to consolidate with, e.g. llama3.2:3b."),
    upstream: str = typer.Option(DEFAULT_UPSTREAM, help="OpenAI-compatible base URL of your local runtime."),
    space: str = typer.Option("default", help="Memory space."),
    host: str = typer.Option("127.0.0.1", help="Interface to look for running Orma proxy."),
    port: int = typer.Option(8080, help="Port of active Orma proxy."),
):
    """Consolidate pending records into states and events now."""
    # Fast path: If Orma proxy is actively running, delegate to its running server instance
    active_url = _get_active_proxy_url(host, port)
    if active_url and model is None:
        try:
            req = urllib.request.Request(
                f"{active_url}/_orma/sleep?space={space}",
                data=b"",
                headers={"User-Agent": "orma-cli"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    typer.echo(f"Consolidated via active Orma proxy on port {port}: "
                               f"+{data.get('states_added', 0)} states, "
                               f"+{data.get('events_added', 0)} events "
                               f"({data.get('episodes_read', 0)} episodes read).")
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            typer.echo(f"Proxy sleep delegation failed ({exc}), falling back to direct consolidation...")

    # Fallback / Standalone path: requires model specification
    target_model = model or os.environ.get("ORMA_MODEL")
    if not target_model:
        typer.echo("Error: Missing --model option and no active Orma proxy was found on port 8080.")
        typer.echo("\nTo fix this, either:")
        typer.echo("  1. Run with explicit model:  orma sleep --model <model_name>")
        typer.echo("  2. Start the proxy first:     orma up --upstream <url> --model <model_name>")
        raise typer.Exit(1)

    memory = _memory(upstream, target_model)
    try:
        report = memory.sleep(space)
        typer.echo(_report(report))
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Sleep consolidation failed: {exc}")
        raise typer.Exit(1)
    finally:
        memory.close()


@app.command()
def rebuild(
    model: str | None = typer.Option(None, help="Model to read everything with, e.g. a stronger one than chat uses."),
    upstream: str = typer.Option(DEFAULT_UPSTREAM, help="OpenAI-compatible base URL of your local runtime."),
    space: str = typer.Option("default", help="Memory space."),
):
    """Forget all states and events and read every kept record again (records themselves are never lost)."""
    target_model = model or os.environ.get("ORMA_MODEL")
    if not target_model:
        typer.echo("Error: Missing --model option.")
        typer.echo("Usage: orma rebuild --model <model_name>")
        raise typer.Exit(1)

    memory = _memory(upstream, target_model)
    try:
        report = memory.rebuild(space)
        typer.echo(_report(report))
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Rebuild failed: {exc}")
        raise typer.Exit(1)
    finally:
        memory.close()


@app.command()
def facts(
    entity: str | None = typer.Argument(None, help="Only states of this entity, e.g. user."),
    space: str = typer.Option("default", help="Memory space."),
    history: bool = typer.Option(False, "--history", help="Include states that are no longer true."),
):
    """List remembered states, each with the sentence that stated it."""
    memory = _memory()
    try:
        tz = clock.zone(memory.config.timezone)
        for state in memory.facts(entity, space, include_closed=history):
            note = "" if state.valid_to is None else f"  (until {clock.iso_day(state.valid_to, tz)})"
            typer.echo(f"[{state.id}] {state.text}{note}")
            for sentence in memory.store.sentences([state.sentence_id]):
                typer.echo(f"      {sentence.label(tz)}")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Error fetching facts: {exc}")
        raise typer.Exit(1)
    finally:
        memory.close()


@app.command()
def recall(
    query: str = typer.Argument(..., help="A message, as a user would send it."),
    space: str = typer.Option("default", help="Memory space."),
    budget: int | None = typer.Option(None, help="Token budget (default from config)."),
):
    """Show the memory block that would be added to a chat prompt for QUERY."""
    if not query or not query.strip():
        typer.echo("Error: query cannot be empty.")
        raise typer.Exit(1)

    memory = _memory()
    try:
        context = memory.build_context([{"role": "user", "content": query}], space, budget)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Recall failed: {exc}")
        raise typer.Exit(1)
    finally:
        memory.close()
    typer.echo(context[0]["content"] if context[0]["role"] == "system" else "(nothing recalled)")


@app.command()
def forget(
    state_id: int = typer.Argument(..., help="State id, as shown by `orma facts`."),
    space: str = typer.Option("default", help="Memory space."),
):
    """Mark a state as no longer true."""
    memory = _memory()
    try:
        forgotten = memory.forget(state_id, space)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Forget failed: {exc}")
        raise typer.Exit(1)
    finally:
        memory.close()
    typer.echo("Forgotten." if forgotten else "No current state with that id in this space.")
    raise typer.Exit(0 if forgotten else 1)


@app.command()
def stats(space: str = typer.Option("default", help="Memory space.")):
    """Counts of what is stored, as JSON."""
    memory = _memory()
    try:
        data = {"db": str(memory.config.db_path), "space": space, **memory.stats(space)}
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Stats failed: {exc}")
        raise typer.Exit(1)
    finally:
        memory.close()
    typer.echo(json.dumps(data, indent=2))


@app.command()
def mcp(
    space: str = typer.Option("default", help="Default memory space for the MCP session."),
    upstream: str | None = typer.Option(None, help="Upstream LLM for background sleep consolidation."),
    model: str | None = typer.Option(None, help="Model name for background sleep consolidation."),
):
    """Run Orma as a standard Model Context Protocol (MCP) server over stdio for coding assistants."""
    from orma.serve.mcp import run_mcp_server
    memory = _memory(upstream=upstream, model=model)
    try:
        run_mcp_server(memory, default_space=space)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"MCP server failed: {exc}")
        raise typer.Exit(1)
    finally:
        memory.close()


if __name__ == "__main__":
    app()
