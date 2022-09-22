"""

[Package Documentation](../blocksim/index.html)

.. include:: README.md

"""
import sys
import os
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import MarkdownExporter


def _bsprint(*args, **kwargs):
    return
    print(*args, **kwargs)


def __f1_newer_than_f2(f1, f2):
    if not os.path.exists(f2):
        return True
    f1time = os.path.getmtime(f1)
    f2time = os.path.getmtime(f2)

    return f1time > f2time


def __read_notebook(path):
    with open(path, "r") as f:
        buf = f.read()
    if len(buf) == 0:
        return None
    nb = nbformat.reads(buf, as_version=4)
    return nb


def __execute_notebook(root, ep, nb):
    ep.preprocess(nb, {"metadata": {"path": root}})


def __render_notebook(exporter, fic, nb, md_pth):
    (body, resources) = exporter.from_notebook_node(nb)
    _bsprint("   ", md_pth)
    rt = os.path.basename(fic).replace(".ipynb", "")
    with open(md_pth, "w") as f:
        f.write(body.replace("![png](", f"![png]({rt}_"))

    imgdir = Path("build/htmldoc") / "examples"
    imgdir.mkdir(parents=True, exist_ok=True)
    for pth_img in resources["outputs"].keys():
        bn = rt + "_" + pth_img
        _bsprint("   ", imgdir / bn)
        f = open(imgdir / bn, "wb")
        f.write(resources["outputs"][pth_img])
        f.close()


def __create_py(src, dst):
    rt = os.path.basename(src).replace(".ipynb", "")
    _bsprint("   ", dst)
    with open(dst, "w") as f:
        f.write(f'"""\n.. include:: ../build/htmldoc/{rt}.md\n"""')


exporter = MarkdownExporter()
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

odir = Path("build") / "htmldoc"
odir.mkdir(parents=True, exist_ok=True)

# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html

# root = "examples"
# lnb=__list_notebooks(root)
def __process(ipynb_pth):
    root = os.path.dirname(ipynb_pth)
    py_pth = Path(root) / os.path.basename(ipynb_pth).replace(".ipynb", ".py")
    md_pth = odir / os.path.basename(fic).replace(".ipynb", ".md")
    if not __f1_newer_than_f2(ipynb_pth, py_pth) and not __f1_newer_than_f2(ipynb_pth, md_pth):
        return

    print(f"Processing {ipynb_pth.name}")
    nb = __read_notebook(ipynb_pth)
    if nb is None:
        return
    __execute_notebook(root, ep, nb)
    __render_notebook(exporter, ipynb_pth, nb, md_pth)
    __create_py(ipynb_pth, py_pth)


# for fic in lnb:
#     __process(fic)
if __name__ == "__main__":
    root = Path(__file__).parent
    for fic in root.glob("*.ipynb"):
        __process(fic)
