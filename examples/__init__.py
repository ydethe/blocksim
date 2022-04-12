"""

[Package Documentation](../blocksim/index.html)

.. include:: README.md

"""
import os
from pathlib import Path
from uuid import uuid4

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import MarkdownExporter
from traitlets.config import Config


c = Config()
# c.MarkdownExporter.preprocessors = ["nbconvert.preprocessors.ExtractOutputPreprocessor"]
exporter = MarkdownExporter(config=c)
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html


files_with_full_path = [
    f.path for f in os.scandir("examples") if f.is_file() and f.path.endswith(".ipynb")
]

for fic in files_with_full_path:
    print(fic)
    with open(fic, "r") as f:
        nb = nbformat.reads(f.read(), as_version=4)
    ep.preprocess(nb, {"metadata": {"path": "./"}})

    odir = Path("build") / "htmldoc"
    odir.mkdir(parents=True, exist_ok=True)

    rt = os.path.basename(fic).replace(".ipynb", "")
    # c.ExtractOutputPreprocessor.output_filename_template = (
    #     rt + "_{cell_index}_{index}{extension}"
    # )
    exporter = MarkdownExporter(config=c)
    (body, resources) = exporter.from_notebook_node(nb)

    pth_dst = odir / os.path.basename(fic).replace(".ipynb", ".md")
    print("   ", pth_dst)
    with open(pth_dst, "w") as f:
        f.write(body.replace("![png](", f"![png]({rt}_"))

    odir = Path("htmldoc") / "examples"
    odir.mkdir(parents=True, exist_ok=True)
    for pth_img in resources["outputs"].keys():
        print("   ", odir / (rt + "_" + pth_img))
        f = open(odir / (rt + "_" + pth_img), "wb")
        f.write(resources["outputs"][pth_img])
        f.close()
