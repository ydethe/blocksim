# https://pypi.org/project/eastereig/
pdoc3 --html --force --config latex_math=True blocksim
pyreverse -s0 blocksim -m yes -f ALL # In pylint package
dot -Tpng classes.dot -o classes.png
