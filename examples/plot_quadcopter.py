r"""
Quadcopter control
==================

"""
###############################################################################
# Main libraries import
# ---------------------

from collections import OrderedDict

import numpy as np

from blocksim.control.System import ASystem, G6DOFSystem
from blocksim.control.Controller import (
    AController,
    LQRegulator,
    AntiWindupPIDController,
)
from blocksim.Simulation import Simulation
from blocksim.control.SetPoint import Step, Rectangular
from blocksim.utils import quat_to_matrix, quat_to_euler
from blocksim.control.Route import Group, Split
from blocksim.Quadcopter.Quadri import Quadri
from blocksim.Quadcopter.AttPilot import AttPilot
from blocksim.Quadcopter.Motor import Motor

###############################################################################
# Construction of the simulation
# ------------------------------

sim = Simulation()

###############################################################################
# We define 4 identical motors, and their associated controllers

for nmot in range(4):
    mot = Motor(nmot)

    tau = 50e-3
    Ks = 0.0

    ctl_mot = AntiWindupPIDController(
        "ctlmot%i" % nmot, snames=["u"], shape_estimation=(2,)
    )
    ctl_mot.D = 0.0
    ctl_mot.I = mot.km / tau
    ctl_mot.P = ctl_mot.I * mot.Jr * mot.R / mot.km ** 2
    ctl_mot.Ks = Ks
    ctl_mot.Umin = -mot.Umax
    ctl_mot.Umax = mot.Umax

    sim.addComputer(mot)
    sim.addComputer(ctl_mot)

###############################################################################
# We define a node that groups the output of the 4 motors
# into a single vector that is used as input by the quadcopter model

grp_inp = OrderedDict()
grp_inp["in0"] = (1,)
grp_inp["in1"] = (1,)
grp_inp["in2"] = (1,)
grp_inp["in3"] = (1,)
grp = Group(
    "grp",
    inputs=grp_inp,
    snames=["gs0", "gs1", "gs2", "gs3"],
)

sim.addComputer(grp)

###############################################################################
# We create an instance of the quadcopter model
# with its initial state

sys = Quadri(mot)
x0 = sys.getInitialStateForOutput("state")
w0 = np.array([2, -1, 3]) / 2
x0[10:13] = w0
sys.setInitialStateForOutput(x0, "state")

sim.addComputer(sys)

###############################################################################
# We generate commands in roll, pitch, yaw

A0 = sys.g
stp = Rectangular("stp", snames=["r", "p", "y", "A"])
stp.doors = np.array(
    [
        (10, np.pi / 4, 0, 20),
        (30, np.pi / 4, 0, 40),
        (50, np.pi / 4, 0, 60),
        (-1, A0, A0, 71),
    ]
)

sim.addComputer(stp)

###############################################################################
# We create an instance of the main attitude controller

ctl = AttPilot("ctlatt", sys, mot)

sim.addComputer(ctl)

###############################################################################
# We define a node that splits the output of the main attitude controller
# into 4 signals, one for each of the motor controller

spt_otp = OrderedDict()
spt_otp["u0"] = (0,)
spt_otp["u1"] = (1,)
spt_otp["u2"] = (2,)
spt_otp["u3"] = (3,)
spt = Split(
    name="spt",
    signal_shape=(7,),
    outputs=spt_otp,
)

sim.addComputer(spt)

###############################################################################
# We connect all the nodes

sim.connect("stp.setpoint", "ctlatt.setpoint")
sim.connect("ctlatt.command", "spt.signal")
sim.connect("spt.u0", "ctlmot0.setpoint")
sim.connect("spt.u1", "ctlmot1.setpoint")
sim.connect("spt.u2", "ctlmot2.setpoint")
sim.connect("spt.u3", "ctlmot3.setpoint")
sim.connect("ctlmot0.command", "mot0.command")
sim.connect("ctlmot1.command", "mot1.command")
sim.connect("ctlmot2.command", "mot2.command")
sim.connect("ctlmot3.command", "mot3.command")
sim.connect("mot0.vel", "ctlmot0.estimation")
sim.connect("mot1.vel", "ctlmot1.estimation")
sim.connect("mot2.vel", "ctlmot2.estimation")
sim.connect("mot3.vel", "ctlmot3.estimation")
sim.connect("mot0.state", "grp.in0")
sim.connect("mot1.state", "grp.in1")
sim.connect("mot2.state", "grp.in2")
sim.connect("mot3.state", "grp.in3")
sim.connect("grp.grouped", "sys.command")
sim.connect("sys.state", "ctlatt.estimation")
sim.connect("sys.euler", "ctlatt.euler")


###############################################################################
# Simulation
# ----------

from blocksim.Graphics import plotVerif
from matplotlib import pyplot as plt

tps = np.arange(0, 70, 0.05)
sim.simulate(tps, progress_bar=False)
log = sim.getLogger()

plotVerif(
    log,
    "Figure 1",
    [{"var": "deg(sys_euler_roll)"}, {"var": "deg(stp_setpoint_r)"}],
    [{"var": "deg(sys_euler_pitch)"}, {"var": "deg(stp_setpoint_p)"}],
    [{"var": "deg(sys_euler_yaw)"}, {"var": "deg(stp_setpoint_y)"}],
)

plt.show()
