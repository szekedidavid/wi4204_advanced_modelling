import numpy as np


def step_concentration(state, t, dt):
    c     = state.c
    u     = state.u
    D_hat = state.D_hat
    phi   = state.phi
    R     = state.R
    r     = state.grid
    dr    = state.dr
    nr    = state.nr
    c_c   = state.c_c
    t_c   = state.t_c

    CFL_adv  = np.max(np.abs(u) / phi * dt / dr)
    CFL_diff = np.max(D_hat * dt / dr**2)

    if CFL_adv > 1:
        print(f"[c] CFL advection warning: {CFL_adv:.3f} > 1")
    if CFL_diff > 0.5:
        print(f"[c] CFL diffusion warning: {CFL_diff:.3f} > 0.5")

    i = np.arange(1, nr - 1)

    r_plus  = r + 0.5 * dr
    r_minus = r - 0.5 * dr

    phiD = phi * D_hat
    phiD_face = 2 * phiD[:-1] * phiD[1:] / (phiD[:-1] + phiD[1:])

    flux_diff = (r_plus[i]  * phiD_face[i]   * (c[i+1] - c[i])
               - r_minus[i] * phiD_face[i-1] * (c[i]   - c[i-1])) / (r[i] * dr**2)
    flux_adv  = u[i] * (c[i] - c[i-1]) / dr
    reaction  = -(t_c / c_c) * R[i]

    c_new = c.copy()
    c_new[i] = c[i] + dt * ((flux_diff - flux_adv) / phi[i] + reaction)

    c_new[0]  = state.c_inj / c_c
    c_new[-1] = c_new[-2]

    state.c[:] = c_new