"""`python -m orma`: the CLI, for when another `smruti` command shadows this one on PATH."""

from orma.cli import app

app(prog_name="orma")
