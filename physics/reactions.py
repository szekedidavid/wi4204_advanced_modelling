import numpy as np

R_GAS = 8.31451  # J / (mol K)


def kdiss_of_T(T_K, A_arrhenius, Ea):
    """Arrhenius dissolution rate constant.
    T_K : temperature in Kelvin, array (nr,)
    Returns Kdiss [mol m^-2 s^-1], array (nr,)
    """
    return A_arrhenius * np.exp(-Ea / (R_GAS * T_K))


def reaction_rate(c, T_K, species, c_c, C_0):
    """Simplified net precipitation rate S_i = SA * Kdiss * (C/C0 - 1).

    Parameters
    ----------
    c       : dimensionless concentration, array (nr,)
    T_K     : temperature in Kelvin, array (nr,)
    species : dict of species parameters
    c_c     : concentration scale [mol / kgw]

    Returns
    -------
    R : net precipitation rate [mol / kgw / s], array (nr,)
    """
    C     = c * c_c
    Kdiss = kdiss_of_T(T_K, species["A_arrhenius"], species["Ea"])
    SA    = species["SA"]

    return SA * Kdiss * (C / C_0 - 1.0)