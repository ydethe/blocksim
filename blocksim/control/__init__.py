"""The control module allows the simulation of controlled system. It includes for example Kalman filters, LQ regulators, 6 DOF systems, and more.

Example:
>>> import numpy as np
>>> from blocksim.control.System import LTISystem
>>> from blocksim.control.Controller import PIDController
>>> from blocksim.Simulation import Simulation
>>> 
>>> # System's parameters
>>> m = 1.0  # Mass
>>> k = 40.0  # Spring rate
>>> f = 5
>>> # System definition
>>> sys = LTISystem("sys", shape_command=(1,), snames_state=["x", "v"])
>>> sys.matA = np.array([[0, 1], [-k / m, -f / m]])
>>> sys.matB = np.array([[0, 1 / m]]).T
>>> # We set the initial position to -1, and the initial velocity to 0
>>> sys.setInitialStateForOutput(np.array([-1.0, 0.0]), "state")
>>> ctl = PIDController("ctl", shape_estimation=(2,), snames=["u"], coeffs=(1.0, 0.0, 0.0))
>>> sim = Simulation([sys, ctl])
>>> sim.connect("sys.state", "ctl.estimation")
>>> sim.connect("ctl.command", "sys.command")
>>> sim.simulate(np.arange(200) / 100, error_on_unconnected=False)

"""
