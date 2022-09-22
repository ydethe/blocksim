import typer

import importlib.metadata as im
from .tools import gnss_sim


app = typer.Typer()


def version_callback(value: bool):
    if value:
        typer.echo(f"Using blocksim in version {im.version('blocksim')}")
        raise typer.Exit()


@app.callback()
def common(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", callback=version_callback, help="Prints blocksim version"
    ),
):
    pass


app.add_typer(gnss_sim.app, name="gnss", help="GNSS constellation simulation")


def main():
    app()


if __name__ == "__main__":
    main()
