# https://pypi.org/project/eastereig/
rm -rf htmldoc build examples/example_*.py htmlcov /tmp/www/blocksim/coverage
mkdir -p build/htmldoc
mkdir -p htmldoc/examples
python3 examples/__init__.py
pdoc --html --force --config latex_math=True -o htmldoc blocksim examples
pyreverse -s0 blocksim -k --colorized -p blocksim -m no --ignore=exceptions.py,LogFormatter.py,DatabaseModel.py,CSVLogger.py,PickleLogger.py,PsqlLogger.py,XlsLogger.py -d htmldoc # In pylint package
dot -Tpng htmldoc/classes_blocksim.dot -o htmldoc/blocksim/classes.png
cp examples/quadcopter.png htmldoc/examples
# nohup python3 -m http.server 2> error.log &
cp -r htmldoc/* /tmp/www/blocksim
python3 -m pytest -n 8 --mpl --mpl-generate-summary=html --mpl-baseline-path=tests/baseline --mpl-results-path=results --cov blocksim tests --doctest-modules blocksim
coverage html
cp -r htmlcov /tmp/www/blocksim/coverage
