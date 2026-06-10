import numpy as np

R_GAS = 8.31451  # J / (mol K)


def kdiss_of_T(T_K, A_arrhenius, Ea):
    return A_arrhenius * np.exp(-Ea / (R_GAS * T_K))


def reaction_rate(c, T_K, precipitates, solute_index, c_c):
    """Net precipitation rate for all precipitates.

    Parameters
    ----------
    c             : dimensionless concentration, array (n_solutes, nr)
    T_K           : temperature in Kelvin, array (nr,)
    precipitates  : list of precipitate dicts (from load_precipitates)
    solute_index  : dict mapping solute name -> row index in c
    c_c           : concentration scale [mol / kgw]

    Returns
    -------
    R : array (n_precip, nr)
    """
    nr     = c.shape[1]
    n_prec = len(precipitates)
    R      = np.zeros((n_prec, nr))

    C = c * c_c  # (n_solutes, nr)

    for i, prec in enumerate(precipitates):
        Kdiss   = kdiss_of_T(T_K, prec["A_arrhenius"], prec["Ea"])
        c_eff   = np.ones(nr) * np.inf
        for sname, nu in prec["nu"].items():
            if nu == 0:
                continue
            j     = solute_index[sname]
            c_eff = np.minimum(c_eff, C[j])

        R[i] = prec["SA"] * Kdiss * (c_eff - 1.0)

    return R