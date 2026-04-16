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

    Pe        = np.max(np.abs(u) * dr / D_hat)
    CFL_adv   = np.max(np.abs(u) * dt / dr)
    CFL_diff  = np.max(D_hat * dt / dr**2)

    if Pe > 2:
        print(f"[c] Peclet warning: Pe = {Pe:.3f} > 2")
    if CFL_adv > 1:
        print(f"[c] CFL advection warning: {CFL_adv:.3f} > 1")
    if CFL_diff > 0.5:
        print(f"[c] CFL diffusion warning: {CFL_diff:.3f} > 0.5")

    # --- FVM face values ---
    # face radii
    r_plus  = r + 0.5 * dr   # (nr,)  r_{i+1/2}
    r_minus = r - 0.5 * dr   # (nr,)  r_{i-1/2}

    # harmonic mean diffusivity at faces (interior)
    D_plus  = 2 * D_hat[:-1] * D_hat[1:]  / (D_hat[:-1] + D_hat[1:])   # (nr-1,)
    D_minus = 2 * D_hat[1:]  * D_hat[:-1] / (D_hat[1:]  + D_hat[:-1])  # (nr-1,) same but offset

    # upwind face velocity
    u_plus  = 0.5 * (u[:-1] + u[1:])   # (nr-1,)  u_{i+1/2}

    # --- assemble RHS (explicit, forward Euler) ---
    c_new = c.copy()

    # interior cells i = 1 .. nr-2
    i = np.arange(1, nr - 1)

    flux_diff_plus  =  r_plus[i]  * D_plus[i-1]  * (c[i+1] - c[i])   / (dr * r[i] * dr)
    flux_diff_minus = -r_minus[i] * D_minus[i-1] * (c[i]   - c[i-1]) / (dr * r[i] * dr)

    # upwind advection
    u_f_plus  = u_plus[i-1]   # u_{i+1/2}
    u_f_minus = u_plus[i-2] if i[0] > 1 else u[0]   # u_{i-1/2}, fallback at i=1

    # vectorised upwind
    u_f_plus_arr  = 0.5 * (u[i] + u[i+1])
    u_f_minus_arr = 0.5 * (u[i] + u[i-1])

    adv_plus  = r_plus[i]  * np.where(u_f_plus_arr  >= 0, c[i],   c[i+1]) * u_f_plus_arr
    adv_minus = r_minus[i] * np.where(u_f_minus_arr >= 0, c[i-1], c[i])   * u_f_minus_arr

    flux_adv = -(adv_plus - adv_minus) / (r[i] * dr)

    reaction = -(t_c / c_c) * R[i]

    c_new[i] = c[i] + dt * (flux_diff_plus + flux_diff_minus + flux_adv / phi[i] + reaction)

    # --- inner BC: Danckwerts (Robin) ---
    # phi * D * dc/dr|_inner = u_inj * (c[0] - c_inj)
    c_inj = state.c_inj / c_c
    c_new[0] = c_inj + (phi[0] * D_hat[0] / (state.u_inj * dr)) * (c[1] - c[0])

    # --- outer BC: homogeneous Neumann ---
    c_new[-1] = c_new[-2]

    state.c[:] = c_new