from typing import Iterable

import numpy as np
import scipy.linalg as lin

from ..control.Controller import LQRegulator
from ..control.System import ASystem


class VTOLPilot(LQRegulator):
    """Outter loop position / velocity controller

    Attributes:
        grav: Gravitation constant (m/s²)
        pitch_d_max: Pitch security : the controller forbids a pitch setpoint of more than pitch_d_max (rad)
        roll_d_max: Roll security : the controller forbids a roll setpoint of more than roll_d_max (rad)

    Args:
        name: name of the VTOLPilot
        grav: Gravitation constant (m/s²)

    """

    def __init__(self, name: str, grav: float):
        LQRegulator.__init__(
            self,
            name=name,
            shape_setpoint=(4,),
            shape_estimation=(6,),
            snames=[
                "fx",
                "fy",
                "fz",
            ],
        )
        self.defineOutput(
            name="att",
            snames=[
                "roll",
                "pitch",
                "yaw",
                "A",
            ],
            dtype=np.float64,
        )
        self.createParameter(name="grav", value=grav)
        self.createParameter(name="pitch_d_max", value=np.pi / 180 * 45)
        self.createParameter(name="roll_d_max", value=np.pi / 180 * 45)

    def update(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
        estimation: np.array,
        att: np.array,
        command: np.array,
    ) -> dict:
        pos_d = setpoint[:3]
        yaw_d = setpoint[3]
        A0 = self.matN @ pos_d - self.matK @ estimation

        A = A0 + np.array([0, 0, self.grav])
        A_cons = lin.norm(A)
        Tx, Ty, Tz = A / A_cons

        if Tz < 1e-3:
            pitch_d = 0.0
        else:
            pitch_d = np.arctan((Ty * np.sin(yaw_d) + np.cos(yaw_d) * Tx) / Tz)
        sr = Tx * np.sin(yaw_d) - Ty * np.cos(yaw_d)
        cr = (
            np.sin(pitch_d) * (Ty * np.sin(yaw_d) + np.cos(yaw_d) * Tx)
            + np.cos(pitch_d) * Tz
        )
        roll_d = np.arctan2(sr, cr)

        roll_d = np.clip(roll_d, -self.roll_d_max, self.roll_d_max)
        pitch_d = np.clip(pitch_d, -self.pitch_d_max, self.pitch_d_max)

        outputs = {}
        outputs["command"] = A0
        outputs["att"] = np.array([roll_d, pitch_d, yaw_d, A_cons])

        return outputs
