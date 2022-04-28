Q=@

PYTEST=$(shell find tests -name "*.py")
PYPKG=$(shell find blocksim -name "*.py")
PYNB=$(wildcard examples/*.ipynb)
PYNBPY = $(PYNB:.ipynb=.py)
DEST?=/tmp/www/blocksim
BLD_DIR=build/htmldoc
BS_VER=$(shell python -c "import blocksim;print(blocksim.__version__)" | tail -n1)

default: doc examples $(BLD_DIR)/coverage/index.html dist/blocksim-$(BS_VER).tar.gz

.PHONY: install doc examples

.coverage: $(PYPKG) $(PYTEST)
	$(Q)echo "Running tests"
	$(Q)test -d $(BLD_DIR)/tests || mkdir -p $(BLD_DIR)/tests
	$(Q)python3 -m pytest --html=$(BLD_DIR)/tests/report.html --self-contained-html --mpl --mpl-generate-summary=html --mpl-results-always --mpl-baseline-path=tests/baseline --mpl-results-path=$(BLD_DIR)/tests/results --cov blocksim tests --doctest-modules blocksim

$(BLD_DIR)/coverage/index.html: .coverage
	$(Q)echo "Generating HTML coverage report"
	$(Q)test -d $(BLD_DIR)/coverage || mkdir -p $(BLD_DIR)/coverage
	$(Q)coverage html -d $(BLD_DIR)/coverage

%.py: %.ipynb
	$(Q)echo "Generating $@"
	$(Q)python3 examples/__init__.py $<

examples: $(PYNBPY)
	$(Q)test -d $(BLD_DIR)/examples || mkdir -p $(BLD_DIR)/examples
	$(Q)cp examples/quadcopter.png $(BLD_DIR)/examples

doc: $(PYNBPY) $(PYPKG)
	$(Q)echo "Generating documentation"
	$(Q)test -d $(BLD_DIR)/blocksim || mkdir -p $(BLD_DIR)/blocksim
	$(Q)pdoc --html --force --config latex_math=True -o $(BLD_DIR) blocksim examples

classes:
	$(Q)pyreverse -s0 blocksim -k --colorized -p blocksim -m no --ignore=exceptions.py,LogFormatter.py,DatabaseModel.py,CSVLogger.py,PickleLogger.py,PsqlLogger.py,XlsLogger.py -d build/htmldoc # In pylint package
	$(Q)dot -Tpng build/htmldoc/classes_blocksim.dot -o $(BLD_DIR)/blocksim/classes.png

dist/blocksim-$(BS_VER).tar.gz: $(PYPKG)
	$(Q)echo "Buliding distribution"
	$(Q)rm -rf dist
	$(Q)python3 setup.py sdist
	$(Q)pip3 wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz

install: doc examples $(BLD_DIR)/coverage/index.html dist/blocksim-$(BS_VER).tar.gz
	$(Q)echo "Installing files"
	$(Q)test -d $(DEST)/dist || mkdir -p $(DEST)/dist
	$(Q)cp -r $(BLD_DIR)/* $(DEST)
	$(Q)cp -r dist/* $(DEST)/dist
