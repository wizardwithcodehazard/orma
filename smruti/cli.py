"""cli.py - The `smruti` command: run the memory proxy, consolidate, and inspect memory."""

import json
import logging

import typer

from smruti.memory import Memory

DEFAULT_UPSTREAM = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint

app = typer.Typer(help="smruti: local neuro-symbolic long-term memory for small LLMs.",
                  no_args_is_help=True, add_completion=False)


def _memory(upstream: str | None = None, model: str | None = None) -> Memory:
    return Memory(upstream=upstream, model=model)


@app.command()
def proxy(
    upstream: str = typer.Option(DEFAULT_UPSTREAM, help="OpenAI-compatible base URL of your local runtime."),
    host: str = typer.Option("127.0.0.1", help="Interface to listen on."),
    port: int = typer.Option(8080, help="Port to listen on."),
    model: str | None = typer.Option(None, help="Model for sleep consolidation (default: the model clients request)."),
    space: str = typer.Option("default", help="Memory space for requests that name none."),
    sleep_poll: float = typer.Option(30.0, help="Seconds between background sleep checks; 0 disables."),
):
    """Run the memory proxy; point any OpenAI-compatible client at http://HOST:PORT/v1."""
    import uvicorn

    from smruti.serve.proxy import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    memory = _memory(upstream, model)
    typer.echo(f"smruti proxy on http://{host}:{port}/v1 -> {upstream} (memory: {memory.config.db_path})")
    app_ = create_app(memory, upstream, default_space=space, sleep_poll_seconds=sleep_poll or None)
    try:
        uvicorn.run(app_, host=host, port=port, log_level="warning")
    finally:
        memory.close()


@app.command()
def sleep(
    model: str = typer.Option(..., help="Model to consolidate with, e.g. llama3.2:3b."),
    upstream: str = typer.Option(DEFAULT_UPSTREAM, help="OpenAI-compatible base URL of your local runtime."),
    space: str = typer.Option("default", help="Memory space."),
):
    """Consolidate pending turns into long-term memory now."""
    memory = _memory(upstream, model)
    try:
        report = memory.sleep(space)
    finally:
        memory.close()
    typer.echo(f"Consolidated {report.episodes} turns in {report.batches} batches "
               f"({report.failed_batches} failed): {report.consolidation}")


@app.command()
def facts(
    entity: str | None = typer.Argument(None, help="Only facts about this entity, e.g. user."),
    space: str = typer.Option("default", help="Memory space."),
    history: bool = typer.Option(False, "--history", help="Include facts that are no longer true."),
):
    """List remembered facts."""
    memory = _memory()
    try:
        for fact in memory.facts(entity, space, include_closed=history):
            note = "" if fact.valid_to is None else "  (no longer true)"
            typer.echo(f"[{fact.id}] {fact.text}  confidence={fact.confidence:.2f}{note}")
    finally:
        memory.close()


@app.command()
def recall(
    query: str = typer.Argument(..., help="A message, as a user would send it."),
    space: str = typer.Option("default", help="Memory space."),
    budget: int | None = typer.Option(None, help="Token budget (default from config)."),
):
    """Show the memory block that would be added to the prompt for QUERY."""
    memory = _memory()
    try:
        context = memory.build_context([{"role": "user", "content": query}], space, budget)
    finally:
        memory.close()
    typer.echo(context[0]["content"] if context[0]["role"] == "system" else "(nothing recalled)")


@app.command()
def forget(
    fact_id: int = typer.Argument(..., help="Fact id, as shown by `smruti facts`."),
    space: str = typer.Option("default", help="Memory space."),
):
    """Mark a fact as no longer true."""
    memory = _memory()
    try:
        forgotten = memory.forget(fact_id, space)
    finally:
        memory.close()
    typer.echo("Forgotten." if forgotten else "No current fact with that id in this space.")
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
