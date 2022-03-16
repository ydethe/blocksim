#! /bin/sh

git clone https://gitlab.com/manawenuz/blocksim.git
source /opt/conda/etc/profile.d/conda.sh
conda activate bs_env
cd blocksim
python setup.py develop
python -m pytest --mpl-generate-path=tests/baseline tests
cd test/baseline
tar -cf /bl.tar *.png

