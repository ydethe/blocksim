# main.py

import typer


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


@app.command()
def coffee(number: int = typer.Argument(default=1, help="Number of mug to cook")):
    """
    Makes coffee
    """
    res = "I made %i coffee" % number
    typer.echo(res)


def main():
    app()


if __name__ == "__main__":
    main()
