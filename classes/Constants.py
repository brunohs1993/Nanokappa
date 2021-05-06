import scipy.constants as ct

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

class Constants:
    def __init__(self):
        
        # self.hbar = ct.physical_constants['Planck constant over 2 pi in eV s'][0]   # hbar in eV s/rad
        
        self.hbar          = ct.physical_constants['reduced Planck constant in eV s'][0]*1e12   # hbar in eV ps/rad = eV / THz rad
        self.kb            = ct.physical_constants['Boltzmann constant in eV/K'     ][0]        # kb in eV/K
        self.ev_in_J       = ct.physical_constants['electron volt'][0]                          # J/eV
        self.a_in_m        = 1e-10                                                              # m/angs
        self.ps_in_s       = 1e-12                                                              # s/ps
        self.eVpsa2_in_Wm2 = self.ev_in_J/(self.ps_in_s*(self.a_in_m)**2)                       # eV/ps a² ---> W/m²
        self.pi            = ct.pi
