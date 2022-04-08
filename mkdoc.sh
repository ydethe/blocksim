# https://pypi.org/project/eastereig/
mkdir -p htmldoc
pyreverse -s0 blocksim -k --colorized -p blocksim -m no --ignore=exceptions.py,LogFormatter.py,DatabaseModel.py,CSVLogger.py,PickleLogger.py,PsqlLogger.py,XlsLogger.py -d htmldoc # In pylint package
dot -Tpng htmldoc/classes_blocksim.dot -o htmldoc/classes.png
pdoc3 --html --force --config latex_math=True htmldoc/blocksim
