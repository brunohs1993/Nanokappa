# Copyright (C) 2024, Bruno Hartmann da Silva
# License: MIT

import numpy as np
from types import FunctionType
from nanokappa.particles import Particle
from nanokappa.utils import Constants as C


class Phonon(Particle):
    omega: float
    occupation: float
    tau_function: FunctionType

    def scatter(self, T: float):
        n0 = self.occupation_function(T)
        tau = self.tau_function(T)
        self.occupation = n0 + (self.occupation - n0) * np.exp(-self.dt / tau)

    def occupation_function(self, T):
        return 0.5 + 1 / (np.exp(C.hbar * self.omega / (C.kb * T)) - 1)
