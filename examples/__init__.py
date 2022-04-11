"""

.. include:: README.md

"""
import os
from pathlib import Path
import nbformat
from nbconvert import MarkdownExporter
from traitlets.config import Config


c = Config()

# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html


files_with_full_path = [
    f.path for f in os.scandir("examples") if f.is_file() and f.path.endswith(".ipynb")
]

for fic in files_with_full_path:
    print(fic)
    with open(fic, "r") as f:
        nb = nbformat.reads(f.read(), as_version=4)

    odir = Path("build") / "htmldoc"
    odir.mkdir(parents=True, exist_ok=True)
    c.MarkdownExporter.preprocessors = ['nbconvert.preprocessors.ExtractOutputPreprocessor']
    # c.NbConvertApp.output_base = str(odir)
    # c.FilesWriter.build_directory = str(odir)
    exporter = MarkdownExporter(config=c)

    (body, resources) = exporter.from_notebook_node(nb)

    pth_dst = odir / os.path.basename(fic).replace(".ipynb", ".md")
    print("   ", pth_dst)
    with open(pth_dst, "w") as f:
        f.write(body)

    odir = Path("htmldoc") / "examples"
    odir.mkdir(parents=True, exist_ok=True)
    for pth_img in resources["outputs"].keys():
        print("   ", odir / pth_img)
        f = open(odir / pth_img, "wb")
        f.write(resources["outputs"][pth_img])
        f.close()

    del exporter, nb
