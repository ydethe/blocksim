# main.py

from typing import Tuple

import typer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .DatabaseModel import Base, Simulation
from .Logger import Logger


app = typer.Typer()
db_app = typer.Typer()
app.add_typer(db_app, name="db", help="Manage database.")


@app.command()
def header(fic_bin: str):
    """
    Visualize a binary log file's header
    """
    from .Logger import Logger

    log = Logger()
    res = log.getFileHeader(fic_bin)
    typer.echo(res)


@db_app.command()
def init(
    uri: str = typer.Argument(
        ...,
        help="URI of the connection. Example : 'postgresql+psycopg2://postgres@localhost/simulations'",
    )
):
    """
    Initializes an empty db
    """
    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.
    engine = create_engine(uri)

    # Create all tables in the engine. This is equivalent to "Create Table"
    # statements in raw SQL.
    Base.metadata.create_all(engine)


@db_app.command()
def list(
    uri: str = typer.Argument(
        ...,
        help="URI of the connection. Example : 'postgresql+psycopg2://postgres@localhost/simulations'",
    )
):
    """
    Lists simulatons in database
    """
    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.
    engine = create_engine(uri)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    q = session.query(Simulation).order_by(Simulation.start_time)
    for r in q.all():
        print(r)


@db_app.command()
def export(
    uri: str = typer.Argument(
        ...,
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
    log.loadLoggerFile("%s?sim_id=%i" % (uri, simid))
    log.export(pth, format)


def main():
    app()


if __name__ == "__main__":
    main()
