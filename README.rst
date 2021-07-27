========
blocksim
========


.. image:: https://img.shields.io/pypi/v/blocksim.svg
        :target: https://pypi.python.org/pypi/blocksim

.. image:: https://readthedocs.org/projects/blocksim/badge/?version=latest
        :target: https://blocksim.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://gitlab.com/manawenuz/blocksim/badges/master/pipeline.svg
   :target: https://gitlab.com/manawenuz/blocksim/pipelines

.. image:: https://codecov.io/gl/manawenuz/blocksim/branch/master/graph/badge.svg
  :target: https://codecov.io/gl/manawenuz/blocksim

.. image:: https://img.shields.io/pypi/dm/blocksim
  :target: https://pypi.python.org/pypi/blocksim


A library to simulate a closed-loop system. Includes :

* Kalman filter
* Mahony and Madgwick estimators
* PID and LQ controllers
* FIR filter
* System simulation
* Sensor simulation (with bias and noise)
* Included quadcopter stabilization example !
* Real time control of arduino based systems via a serial link
* Real time plotting
* yaml file configuration
* Routing functions (group, split, ...)

Free software: MIT license

Development
-----------

With conda::

    conda env create -f environment.yml
    source activate blocksim
    python setup.py develop

Features
--------

* TODO
