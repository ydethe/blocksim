# https://pypi.org/project/eastereig/
mkdir -p htmldoc
pdoc --html --force --config latex_math=True -o htmldoc blocksim
pyreverse -s0 blocksim -k --colorized -p blocksim -m no --ignore=exceptions.py,LogFormatter.py,DatabaseModel.py,CSVLogger.py,PickleLogger.py,PsqlLogger.py,XlsLogger.py -d htmldoc # In pylint package
dot -Tpng htmldoc/classes_blocksim.dot -o htmldoc/blocksim/classes.png
