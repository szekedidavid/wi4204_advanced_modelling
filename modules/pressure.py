import numpy as np


def solve_pressure(state, t):
    r      = state.grid
    k_hat  = state.k_hat
    mu     = state.mu[0]
    u_inj  = state.u_inj
    r_0    = state.r_0

    integrand = 1.0 / (r * k_hat)

    cum = np.zeros_like(r)
    for i in range(len(r) - 1):
        cum[i+1] = cum[i] + 0.5 * (integrand[i] + integrand[i+1]) * (r[i+1] - r[i])

    total = cum[-1]
    state.p[:] = mu * u_inj * r_0 * (total - cum)


def compute_velocity(state, t):
    state.u[:] = state.u_inj * state.r_0 / state.grid