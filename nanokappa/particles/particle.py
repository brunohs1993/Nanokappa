# Copyright (C) 2023-2024, Bruno Hartmann da Silva
# License: MIT

import numpy as np
from nanokappa.base import BaseModel


class Particle(BaseModel):
    """Base particle classe.

    Attributes
    ----------
    position : np.ndarray
        Position of the particle.
    velocity : np.ndarray
        Velocity of the particle.
    """
    position: np.ndarray
    velocity: np.ndarray

    def move(self, dt):
        """Move the particle one timestep.

        Parameters
        ----------
        dt: The time step.
        """
        self.position = self.velocity * dt
