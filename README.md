# blocksim

## Quick look

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

blocksim is hosted here https://git:8443/projects/DNFSND/repos/blocksim/browse

## Setup

Create a virtual environment named bs_env for example. You do not have to create one env per project or per simulation.
But you need one where blocksim will be installed.
See https://realpython.com/python-virtual-environments-a-primer/ and https://www.dataquest.io/blog/a-complete-guide-to-python-virtual-environments/ to learn more about python environments.

To create a virtual env **bs_env** in a given root folder (typically $HOME/.venvs):

    python3 -m venv /path/to/root/folder/bs_env

Then you need to activate the env. This can be done with one of the following options:

1. by modifying the $HOME/.bashrc file (automatically sourced at each login)
1. manually
1. VSCode with the python extension can also associate a project to an env

In the first 2 cases, the line to type is:

    source /path/to/root/folder/bs_env/bin/activate

Once the env is active, clone the repository. Note that the repository can be cloned in any other folder:

    git clone ssh://git@git:7999/dnfsnd/blocksim.git
    cd blocksim

In your virtual env:

    python3 setup.py develop

That's it ! You are now ready to use blocksim library.
In the folder tests and examples are a lot of examples that can be used as a starting point.

## Run tests

To run tests, just run:

    python3 -m pytest --mpl --mpl-generate-summary=html --mpl-baseline-path=tests/baseline --mpl-results-path=results --cov blocksim tests --doctest-modules blocksim

Once the tests are run, the code coverage is available. To have a html version in the htmlcov folder, run:

    coverage html

If needed (for example, a new test with its associated baseline image), we might have to regenerate the baseline images. In this case, run:

    python3 -m pytest --mpl-generate-path=tests/baseline tests

## Build the doc

Just run:

    pdoc --html --force -o htmldoc --config latex_math=True blocksim

A few guidelines for updating the doc
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
