# https://pypi.org/project/eastereig/
pdoc3 --html --force --config latex_math=True blocksim
pyreverse -s0 blocksim -k --colorized -p blocksim -m no --ignore=exceptions.py,LogFormatter.py,DatabaseModel.py,CSVLogger.py,PickleLogger.py,PsqlLogger.py,XlsLogger.py # In pylint package
dot -Tpng classes_blocksim.dot -o classes.png
