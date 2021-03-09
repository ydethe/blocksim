#!/bin/sh

rm -rf logs *.log test-results .tox .eggs public build dist htmlcov
rm -rf docs/_build docs/blocksim.* *.egg-info .pytest_cache
rm -rf tests/config2.yml
find . -name "__pycache__" -exec rm -rf {} \;
find . -name "*.o" -exec rm -rf {} \;
find . -name "*.bak" -exec rm -rf {} \;
rm -rf .DS_Store .coverage
