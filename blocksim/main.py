# main.py

from typing import Tuple
import os

import typer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from . import logger
from .Logger import Logger


app = typer.Typer()


@app.command()
def header(fic_bin: str):
    """
    Visualize a binary log file's header
    """
    from .Logger import Logger

    log = Logger()
    res = log.getFileHeader(fic_bin)
    typer.echo(res)


def main():
    app()


if __name__ == "__main__":
    main()
