[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Python](https://img.shields.io/badge/python-3.8-green)
![Doc](htmldoc/doc_badge.svg)
![Coverage](htmldoc/cov_badge.svg)

# Quick look

A library to simulate a closed-loop system

# Build the doc

Just run:

    git-changelog --output CHANGELOG.md --in-place --version-regex '<a href="[^"]+">(?P<version>[^<]+)' --marker-line '<!-- new entries will be injected here -->'
    pdm doc

This will create the doc in htmldoc

A few guidelines for updating the doc
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Link to the documentation: https://ydethe.gitlab.io/blocksim/htmldoc/blocksim/
