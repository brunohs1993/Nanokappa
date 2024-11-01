# Copyright (C) 2024, Bruno Hartmann da Silva
# License: MIT

import numpy as np
from pydantic import BaseModel


class Particle(BaseModel):
    position: np.ndarray
    velocity: np.ndarray

    def move(self, dt):
        """Move the particle one timestep.

        Parameters
        ----------
        dt: The time step in seconds.
        """
        self.position = self.velocity * dt
