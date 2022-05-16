"""Implementation of entry points for the CLI program

"""

import os

import typer

from .loggers.Logger import Logger
from . import logger


app = typer.Typer()


@app.command()
def header(fic_bin: str):
    """
    Visualize a binary log file's header
    """
    typer.echo("Je suis un poney")


def main():
    app()


if __name__ == "__main__":
    main()
