import os
import nbformat
from nbconvert import MarkdownExporter


# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html

exporter=MarkdownExporter()

files_with_full_path = [f.path for f in os.scandir("examples") if f.is_file()]

for fic in files_with_full_path:
    print(fic)
    with open(fic,'r') as f:
        nb = nbformat.reads(f.read(), as_version=4)

    (body, resources) = exporter.from_notebook_node(nb)

    pth_dst=fic.replace('.ipynb','.md')
    with open(pth_dst,'w') as f:
        f.write(body)
    
    for pth_img in resources['outputs'].keys():
        f = open(os.path.join("htmldoc","examples",pth_img),'wb')
        f.write(resources['outputs'][pth_img])
        f.close()
