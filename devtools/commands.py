import subprocess

from sphinx.cmd.build import main


def build_baseline():
    cmd = ["python", "-m", "pytest", "--mpl-generate-path=tests/baseline", "tests"]
    subprocess.run(cmd)


def build_doc():
    args = [
        "-T",
        "-b",
        "html",
        "-D",
        "language=fr",
        "docs",
        "htmldoc",
    ]
    return main(args)
