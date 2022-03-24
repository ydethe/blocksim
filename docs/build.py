# -*- coding: utf-8 -*-

import re
import sys

from sphinx.cmd.build import main


def bld_doc():
    args = [
        "-T",
        "-b",
        "html",
        "-D",
        "language=fr",
        "docs",
        "htmldoc",
    ]
    main(args)
