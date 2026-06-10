import numpy as np

R_GAS = 8.31451  # J / (mol K)


def kdiss_of_T(T_K, A_arrhenius, Ea):
    return A_arrhenius * np.exp(-Ea / (R_GAS * T_K))


def reaction_rate(c, T_K, precipitates, solute_index, c_c, c_0):
    """Net precipitation rate for all precipitates.

    Parameters
    ----------
    c             : concentration array (n_solutes, nr)
    T_K           : temperature in Kelvin (nr,)
    precipitates  : list of precipitate dicts
    solute_index  : dict mapping solute name -> row index in c
    c_c           : concentration scale [mol / kgw]
    c_0           : initial concentrations (n_solutes,) [mol / kgw]

    Returns
    -------
    R : array (n_precip, nr)  [mol / kgw / s]
    """
    nr    = c.shape[1]
    R     = np.zeros((len(precipitates), nr))
    C     = c * c_c  # (n_solutes, nr)

    for i, prec in enumerate(precipitates):
        Kdiss    = kdiss_of_T(T_K, prec["A_arrhenius"], prec["Ea"])
        c_eff    = np.ones(nr) * np.inf
        c_eff_0  = np.inf
        for sname, nu in prec["nu"].items():
            if nu == 0:
                continue
            j        = solute_index[sname]
            c_eff    = np.minimum(c_eff,   C[j])
            c_eff_0  = min(c_eff_0, c_0[j])

        R[i] = np.maximum(0.0, prec["SA"] * Kdiss * (c_eff / c_eff_0 - 1.0))

    return R