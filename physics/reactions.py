import numpy as np

R_GAS = 8.31451  # J / (mol K)


def keq_of_T(T_K, A1, A2, A3, A4, A5, A6):
    """Equilibrium constant from phenomenological log10 polynomial (eq. 30).
    T_K : temperature in Kelvin, array (nr,)
    Returns Keq, array (nr,)
    """
    log10_keq = (A1
                 + A2 * T_K
                 + A3 / T_K
                 + A4 * np.log10(T_K)
                 + A5 / T_K**2
                 + A6 * T_K**2)
    return 10**log10_keq


def kdiss_of_T(T_K, A_arrhenius, Ea):
    """Arrhenius dissolution rate constant (eq. 30).
    T_K : temperature in Kelvin, array (nr,)
    Returns Kdiss [mol m^-2 s^-1], array (nr,)
    """
    return A_arrhenius * np.exp(-Ea / (R_GAS * T_K))


def reaction_rate(c, T_K, phi, species, c_c):
    """Net precipitation rate S_i (eq. 29) for a single species.

    Parameters
    ----------
    c       : dimensionless concentration, array (nr,)
    T_K     : temperature in Kelvin, array (nr,)
    phi     : porosity, array (nr,)
    species : dict of species parameters from CSV loader
    c_c     : concentration scale [mol / kgw]

    Returns
    -------
    R : net precipitation rate [mol / kgw / s], array (nr,)
    """
    # recover dimensional concentration
    C = c * c_c  # [mol / kgw]

    Kdiss = kdiss_of_T(T_K, species["A_arrhenius"], species["Ea"])
    Keq   = keq_of_T(T_K,
                     species["A1"], species["A2"], species["A3"],
                     species["A4"], species["A5"], species["A6"])

    SA  = species["SA"]
    nu  = species["nu"]
    eta = species["eta"]
    theta = species["theta"]

    # IAP / Keq — single species AB <-> A + B, simplified to C / C_eq
    # with C_eq = Keq (activity coefficient folded into Keq as in paper eq. 51)
    C_eq = Keq  # TODO: extend to full IAP with activity coefficients

    ratio = (C / C_eq)**theta - 1.0

    R = nu * SA * Kdiss * np.sign(ratio) * np.abs(ratio)**eta

    return R