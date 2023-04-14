[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Python](https://img.shields.io/badge/python-3.8%7C3.9%7C3.10-green)
![Doc](../doc_badge.svg)
![Coverage](../cov_badge.svg)

# Quick look

A library to simulate a open- and closed-loop system. Includes :

-   Kalman filter
-   Mahony and Madgwick estimators
-   PID and LQ controllers
-   System simulation
-   Sensor simulation (with bias and noise)
-   Included quadcopter stabilization example !
-   Real time control of arduino based systems via a serial link
-   Real time plotting
-   Routing functions (group, split, \...)
-   Simple GNSS tracking and PVT
-   Satellite propagation (SGP4 or simple circle)
-   Advanced plotting functions
-   DSP tools

blocksim is hosted here https://gitlab.com/ydethe/blocksim

# Setup for developement

First, clone the repository:

    git clone git@gitlab.com:ydethe/blocksim.git
    cd blocksim

Create a virtual env:

    pdm venv create

Tell pdm to use the newly created venv. It should be like `/path/to/current/dir/.venv/bin/python`:

    pdm use

Install the dependencies, and the paquet itself as editable:

    pdm install

That's it ! You are now ready to use blocksim library.
In the folder tests and examples are a lot of examples that can be used as a starting point.

# Build the doc

Just run:

    pdm doc

This will create the doc in build/htmldoc

A few guidelines for updating the doc
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

For now, it is hosted here: https://ydethe.gitlab.io/blocksim/blocksim/
