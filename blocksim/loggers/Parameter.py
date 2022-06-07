from dataclasses import dataclass, field
from typing import List, Any

import numpy as np


@dataclass(init=True)
class Parameter:
    """Represents a parameter of the simulation"""

    name: str
    unit: str
    description: Any
    dtype: Any
    data: List[complex] = field(default_factory=list)

    def __repr__(self):
        if self.description == "":
            res = f"Parameter '{self.name}' ({self.unit})"
        else:
            res = f"Parameter '{self.name}' ({self.unit}): {self.description}"
        return res

    def getTypeDB(self) -> str:
        val = self.dtype()
        if isinstance(val, bytes):
            typ = val
        elif isinstance(val, (int, np.int8, np.int16, np.int32, np.int64)):
            typ = "integer"
        elif isinstance(val, (float, np.float16, np.float32, np.float64)):
            typ = "float"
        elif isinstance(val, (complex, np.complex64, np.complex128)):
            typ = "complex"
        elif isinstance(val, (bool)):
            typ = "boolean"
        else:
            raise ValueError("Impossible to determine type of %s" % val)

        return typ
