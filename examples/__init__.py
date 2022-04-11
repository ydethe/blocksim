import os
import nbformat
from nbconvert import MarkdownExporter


# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html

exporter=MarkdownExporter()

for dirpath,dirnames,filenames in os.walk('examples'):
    for fic in filenames:
        if not fic.endswith('.ipynb'):
            continue

        print(os.path.join(dirpath,fic))
        with open(os.path.join(dirpath,fic),'r') as f:
            nb = nbformat.reads(f.read(), as_version=4)

        (body, resources) = exporter.from_notebook_node(nb)

        pth_dst=fic.replace('.ipynb','.md')
        with open(os.path.join(dirpath,pth_dst),'w') as f:
            f.write(body)
        
        for pth_img in resources['outputs'].keys():
            f = open(os.path.join("htmldoc","examples",pth_img),'wb')
            f.write(resources['outputs'][pth_img])
            f.close()
