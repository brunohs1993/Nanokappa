# Copyright (C) 2024, Bruno Hartmann da Silva
# License: MIT

import scipy.constants as ct
from enum import Enum


class Constants(Enum):
    hbar = ct.physical_constants['reduced Planck constant in eV s'][0] * 1e12  # hbar in eV ps/rad = eV / THz rad
    kb = ct.physical_constants['Boltzmann constant in eV/K'][0]  # kb in eV/K
    ev_in_J = ct.physical_constants['electron volt'][0]  # J/eV
    a_in_m = 1e-10  # m/angs
    ps_in_s = 1e-12  # s/ps
    eVpsa2_in_Wm2 = ev_in_J / (ps_in_s * (a_in_m) ** 2)  # eV/ps a² ---> W/m²
    pi = ct.pi
