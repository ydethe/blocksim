from dataclasses import dataclass, field
from typing import List, Any


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
