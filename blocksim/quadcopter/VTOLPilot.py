import numpy as np
import scipy.linalg as lin

from ..control.Controller import AController
from ..control.System import ASystem


class VTOLPilot(AController):
    """Outter loop position / velocity controller

    Attributes:
        sys: ASystem instance to be controlled
        ctl: AController to use for the control
        pitch_d_max: Pitch security : the controller forbids a pitch setpoint of more than pitch_d_max (rad)
        roll_d_max: Roll security : the controller forbids a roll setpoint of more than roll_d_max (rad)

    Args:
        name: name of the VTOLPilot
        sys: ASystem instance to be controlled
        ctl: AController to use for the control

    """

    def __init__(self, name: str, sys: ASystem, ctl: AController):
        AController.__init__(
            self,
            name=name,
            shape_setpoint=(4,),
            shape_estimation=(6,),
            snames=[
                "fx",
                "fy",
                "fz",
                "roll",
                "pitch",
                "yaw",
                "A",
            ],
        )
        self.createParameter(name="sys", value=sys)
        self.createParameter(name="ctl", value=ctl)
        self.createParameter(name="pitch_d_max", value=np.pi / 180 * 45)
        self.createParameter(name="roll_d_max", value=np.pi / 180 * 45)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
        estimation: np.array,
        command: np.array,
    ) -> dict:
        pos_d = setpoint[:3]
        yaw_d = setpoint[3]
        A0 = self.ctl.matN @ pos_d - self.ctl.matK @ estimation

        A = A0 + np.array([0, 0, self.sys.g])
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

        u = np.hstack((A0, np.array([roll_d, pitch_d, yaw_d, A_cons])))

        outputs = {}
        outputs["command"] = u

        return outputs
