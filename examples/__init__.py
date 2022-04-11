"""
.. include:: example_filtering.md

"""

# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html

import os
import nbformat
from nbconvert import MarkdownExporter



def convert_nb(pth, exporter):
    with open(os.path.join("examples",pth),'r') as f:
        nb = nbformat.reads(f.read(), as_version=4)

    (body, resources) = exporter.from_notebook_node(nb)

    pth_dst=pth.replace('.ipynb','.md')
    with open(os.path.join("examples",pth_dst),'w') as f:
        f.write(body)
    
    for pth_img in resources['outputs'].keys():
        f = open(pth_img,'wb')
        f.write(resources['outputs'][pth_img])
        f.close()

exporter=MarkdownExporter()
convert_nb('example_filtering.ipynb', exporter)

