# Copyright (C) 2023-2024, Bruno Hartmann da Silva
# License: MIT

import numpy as np
from typing import Optional, Literal
from nanokappa.base import BaseModel, VectorList3D
from scipy.interpolate import NearestNDInterpolator


class SubvolClassifier(BaseModel):
    n: int
    xc: Optional[VectorList3D] = None
    a: Optional[Literal[0] | Literal[1] | Literal[2]] = None

    def __init__(self):
        super.__init__()
        if not self.xc:
            self.xc = np.ones((self.n, 3)) * 0.5
            # center positions
            self.xc[:, self.a] = np.linspace(0, 1 - 1 / self.n, self.n) + 1 / (2 * self.n)

        self.f = NearestNDInterpolator(self.xc, np.arange(self.n, dtype=int))

    def predict(self, x):
        return self.f(x).astype(int)
