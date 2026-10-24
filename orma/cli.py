"""cli.py - The `smruti` command: add to memory, search it, consolidate it, inspect it, and run the chat proxy."""

import json
import logging
from pathlib import Path

import typer

from orma import clock
from orma.memory import Memory

DEFAULT_UPSTREAM = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint

app = typer.Typer(help="smruti: local neuro-symbolic long-term memory for small LLMs.",
                  no_args_is_help=True, add_completion=False)


def _memory(upstream: str | None = None, model: str | None = None) -> Memory:
    return Memory(upstream=upstream, model=model)


@app.command()
def add(
    content: str = typer.Argument(..., help="A file path, or the text itself."),
    space: str = typer.Option("default", help="Memory space."),
    role: str | None = typer.Option(None, help="note (your own words) or document (default: document for a "
                                               "file, note for text)."),
):
    """Add a note or a text file to memory; it is searchable at once and consolidated at the next sleep."""
    memory = _memory()
    try:
        path = Path(content)
        if path.is_file():
            results = memory.add_file(path, space, role=role or "document")
        else:
            results = memory.add(content, space, role=role or "note")
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
    memory = _memory()
    try:
        found = memory.search(query, space)
        tz = clock.zone(memory.config.timezone)
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
    memory = _memory(sleep_upstream or upstream, model)
    sleeps_on = f", sleep -> {sleep_upstream}" if sleep_upstream else ""
    ui_url = f"http://{host}:{port}/#/"
    typer.echo(f"orma proxy on http://{host}:{port}/v1 -> {upstream}{sleeps_on} (memory: {memory.config.db_path})")
    typer.echo(f"Web UI available at: {ui_url}")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(ui_url)).start()

    app_ = create_app(memory, upstream, default_space=space, sleep_poll_seconds=sleep_poll or None)
    try:
        uvicorn.run(app_, host=host, port=port, log_level="warning")
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


def _report(report) -> str:
    return (f"Consolidated {report.episodes} records in {report.batches} batches "
            f"({report.failed_batches} failed): {report.consolidation}")


@app.command()
def sleep(
    model: str = typer.Option(..., help="Model to consolidate with, e.g. llama3.2:3b."),
    upstream: str = typer.Option(DEFAULT_UPSTREAM, help="OpenAI-compatible base URL of your local runtime."),
    space: str = typer.Option("default", help="Memory space."),
):
    """Consolidate pending records into states and events now."""
    memory = _memory(upstream, model)
    try:
        report = memory.sleep(space)
    finally:
        memory.close()
    typer.echo(_report(report))


@app.command()
def rebuild(
    model: str = typer.Option(..., help="Model to read everything with, e.g. a stronger one than chat uses."),
    upstream: str = typer.Option(DEFAULT_UPSTREAM, help="OpenAI-compatible base URL of your local runtime."),
    space: str = typer.Option("default", help="Memory space."),
):
    """Forget all states and events and read every kept record again (records themselves are never lost)."""
    memory = _memory(upstream, model)
    try:
        report = memory.rebuild(space)
    finally:
        memory.close()
    typer.echo(_report(report))


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
    finally:
        memory.close()


@app.command()
def recall(
    query: str = typer.Argument(..., help="A message, as a user would send it."),
    space: str = typer.Option("default", help="Memory space."),
    budget: int | None = typer.Option(None, help="Token budget (default from config)."),
):
    """Show the memory block that would be added to a chat prompt for QUERY."""
    memory = _memory()
    try:
        context = memory.build_context([{"role": "user", "content": query}], space, budget)
    finally:
        memory.close()
    typer.echo(context[0]["content"] if context[0]["role"] == "system" else "(nothing recalled)")


@app.command()
def forget(
    state_id: int = typer.Argument(..., help="State id, as shown by `smruti facts`."),
    space: str = typer.Option("default", help="Memory space."),
):
    """Mark a state as no longer true."""
    memory = _memory()
    try:
        forgotten = memory.forget(state_id, space)
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
    finally:
        memory.close()
    typer.echo(json.dumps(data, indent=2))
