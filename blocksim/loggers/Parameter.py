from dataclasses import dataclass, field
from typing import List


@dataclass(init=True)
class Parameter:
    """Represents a parameter of the simulation"""

    name: str
    unit: str
    description: str
    data: List[complex] = field(default_factory=list)

    def __repr__(self):
        if self.description == "":
            res = f"Parameter '{self.name}' ({self.unit})"
        else:
            res = f"Parameter '{self.name}' ({self.unit}): {self.description}"
        return res
