# main.py

from typing import Tuple
import os
import argparse

from . import logger
from .Logger import Logger

import typer


app = typer.Typer()

db_app = typer.Typer()
app.add_typer(db_app, name="db", help="Manage databases.")

def determine_db_uri() -> str:
    """Tries different configuration options to find the database URI.
    First, looks for a `~/.blocksimrc` file
    Then for `BLOCKSIM_DB_URI` environment variable

    Returns:
      The database URI

    """
    pth = os.path.expanduser("~/.blocksimrc")
    if os.path.exists(pth):
        with open(pth, "r") as f:
            uri = f.readline().strip()
    elif "BLOCKSIM_DB_URI" in os.environ.keys():
        uri = os.environ["BLOCKSIM_DB_URI"]
    else:
        uri = None

    return uri

@app.command()
def header(fic_bin: str):
    """
    Visualize a binary log file's header
    """
    typer.echo("Je suis un poney")

@db_app.command()
def init(
    uri: str = typer.Option(
        "",
        help="URI of the connection. Example : 'postgresql+psycopg2://postgres@localhost/simulations'",
    )
):
    """
    Initializes an empty db
    """
    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.
    if uri == "":
        uri = determine_db_uri()
    logger.info("Using '%s'" % uri)
    engine = create_engine(uri)

    # Create all tables in the engine. This is equivalent to "Create Table"
    # statements in raw SQL.
    Base.metadata.create_all(engine)


@db_app.command()
def ls(
    uri: str = typer.Option(
        "",
        help="URI of the connection. Example : 'postgresql+psycopg2://postgres@localhost/simulations'",
    )
):
    """
    Lists simulatons in database
    """
    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.
    if uri == "":
        uri = determine_db_uri()
    logger.info("Using '%s'" % uri)
    engine = create_engine(uri)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    q = session.query(Simulation).order_by(Simulation.start_time)
    for r in q.all():
        print(r)


@db_app.command()
def export(
    uri: str = typer.Option(
        "",
        help="URI of the connection. Example : 'postgresql+psycopg2://postgres@localhost/simulations'",
    ),
    simid: int = typer.Argument(..., help="simulation id"),
    pth: str = typer.Argument(..., help="file to be written"),
    format: str = typer.Option("", help="Format of the file."),
):
    """
    Exports simulation data in a file
    """
    log = Logger()
    if uri == "":
        uri = determine_db_uri()
    logger.info("Using '%s'" % uri)
    log.loadLogFile("%s?sim_id=%i" % (uri, simid))
    log.export(pth, format)

def main():
    app()


if __name__ == "__main__":
    main()
