#!/bin/sh

tar --exclude=*.png --exclude=*.pyc --exclude=blocksim*.rst --exclude=_build --exclude=.vscode --exclude=docs/modules.rst --exclude=docs/examples --exclude=results -cazf ../bs.tar.bz2 . 
