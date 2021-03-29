import scipy.constants as ct

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

class Constants:
    def __init__(self):
        
        # self.hbar = ct.physical_constants['Planck constant over 2 pi in eV s'][0]   # hbar in eV s
        
        self.hbar    = ct.physical_constants['reduced Planck constant in eV s'  ][0]   # hbar in eVs
        self.kb      = ct.physical_constants['Boltzmann constant in eV/K'       ][0]   # kb in eV/K
        self.ev_in_J = ct.physical_constants['electron volt'][0]                     # J/eV
        self.pi      = ct.pi
