import numpy as np


def step_concentration(state, t, dt):
    c     = state.c        # (n_solutes, nr)
    u     = state.u        # (nr,)
    D_hat = state.D_hat    # (n_solutes,)
    phi   = state.phi      # (nr,)
    R     = state.R        # (n_precip, nr)
    nu    = state.nu       # (n_precip, n_solutes)
    r     = state.grid
    dr    = state.dr
    nr    = state.nr
    c_c   = state.c_c
    t_c   = state.t_c

    i = np.arange(1, nr - 1)
    r_plus  = r + 0.5 * dr
    r_minus = r - 0.5 * dr

    # reaction sink per solute: (n_solutes, nr) = nu^T @ R
    dc_rxn = nu.T @ R  # (n_solutes, nr)

    c_new = c.copy()

    for s in range(state.n_solutes):
        D_s = D_hat[s]

        CFL_adv  = np.max(np.abs(u) / phi * dt / dr)
        CFL_diff = np.max(D_s * dt / dr**2)
        if CFL_adv > 1:
            print(f"[c] solute {s} CFL advection warning: {CFL_adv:.3f}")
        if CFL_diff > 0.5:
            print(f"[c] solute {s} CFL diffusion warning: {CFL_diff:.3f}")

        phiD      = phi * D_s
        denom     = phiD[:-1] + phiD[1:]
        phiD_face = np.where(denom > 0, 2 * phiD[:-1] * phiD[1:] / denom, 0.0)

        flux_diff = (r_plus[i]  * phiD_face[i]   * (c[s, i+1] - c[s, i])
                   - r_minus[i] * phiD_face[i-1] * (c[s, i]   - c[s, i-1])) / (r[i] * dr**2)
        flux_adv  = u[i] * (c[s, i] - c[s, i-1]) / dr
        reaction  = -(t_c / c_c) * dc_rxn[s, i]

        c_new[s, i] = c[s, i] + dt * ((flux_diff - flux_adv) / phi[i] + reaction)
        c_new[s, 0]  = state.c_inj[s] / c_c
        c_new[s, -1] = c_new[s, -2]

    state.c[:] = c_new