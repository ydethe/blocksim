import pkg_resources

import typer


app = typer.Typer()


@app.command()
def main(fic: str, dev: bool = True):
    env = dict(tuple(str(ws).split()) for ws in pkg_resources.working_set)
    with open(fic, "r") as f:
        lines = f.readlines()
    for l in lines:
        ls = l.strip()
        if "=" in l:
            print(ls)
        else:
            if ls in env.keys():
                print(f"{ls}=={env[ls]}")
            else:
                print(f"{ls}==")


def main():
    app()


if __name__ == "__main__":
    main()
