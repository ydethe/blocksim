"""

[Package Documentation](../blocksim/index.html)

.. include:: README.md

"""
import os
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import MarkdownExporter


def __list_notebooks(root):
    files_with_full_path = (
        f.path for f in os.scandir(root) if f.is_file() and f.path.endswith(".ipynb")
    )
    return files_with_full_path


def __read_notebook(path):
    with open(path, "r") as f:
        nb = nbformat.reads(f.read(), as_version=4)
    return nb


def __execute_notebook(ep, nb):
    ep.preprocess(nb, {"metadata": {"path": "./"}})


def __render_notebook(exporter, nb, odir):
    (body, resources) = exporter.from_notebook_node(nb)
    pth_dst = odir / os.path.basename(fic).replace(".ipynb", ".md")
    print("   ", pth_dst)
    rt = os.path.basename(fic).replace(".ipynb", "")
    with open(pth_dst, "w") as f:
        f.write(body.replace("![png](", f"![png]({rt}_"))

    imgdir = Path("htmldoc") / "examples"
    imgdir.mkdir(parents=True, exist_ok=True)
    for pth_img in resources["outputs"].keys():
        bn = rt + "_" + pth_img
        print("   ", imgdir / bn)
        f = open(imgdir / bn, "wb")
        f.write(resources["outputs"][pth_img])
        f.close()


def __create_py(root, fic):
    pth_py = Path(root) / os.path.basename(fic).replace(".ipynb", ".py")
    rt = os.path.basename(fic).replace(".ipynb", "")
    with open(pth_py, "w") as f:
        f.write(f'"""\n.. include:: ../build/htmldoc/{rt}.md\n"""')


exporter = MarkdownExporter()
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

odir = Path("build") / "htmldoc"
odir.mkdir(parents=True, exist_ok=True)

# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html

for fic in __list_notebooks("examples"):
    print(fic)
    nb = __read_notebook(fic)
    __execute_notebook(ep, nb)
    __render_notebook(exporter, nb, odir)
    __create_py("examples", fic)
