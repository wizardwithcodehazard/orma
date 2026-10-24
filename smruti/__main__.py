"""`python -m smruti`: the CLI, for when another `smruti` command shadows this one on PATH."""

from smruti.cli import app

app(prog_name="smruti")
