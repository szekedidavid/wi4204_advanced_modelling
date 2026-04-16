import numpy as np


def step_heat(state, t, dt):
    T         = state.T
    u         = state.u
    alpha_hat = state.alpha_hat
    gamma     = state.gamma
    r         = state.grid
    dr        = state.dr
    nr        = state.nr

    Pe       = np.max(np.abs(u) * dr / alpha_hat)
    CFL_adv  = np.max(np.abs(u) * dt / dr)
    CFL_diff = np.max(alpha_hat * dt / dr**2)

    if Pe > 2:
        print(f"[T] Peclet warning: Pe = {Pe:.3f} > 2")
    if CFL_adv > 1:
        print(f"[T] CFL advection warning: {CFL_adv:.3f} > 1")
    if CFL_diff > 0.5:
        print(f"[T] CFL diffusion warning: {CFL_diff:.3f} > 0.5")

    # --- FVM face radii ---
    r_plus  = r + 0.5 * dr
    r_minus = r - 0.5 * dr

    # harmonic mean diffusivity at faces
    a_plus = 2 * alpha_hat[:-1] * alpha_hat[1:] / (alpha_hat[:-1] + alpha_hat[1:])  # (nr-1,)

    T_new = T.copy()

    # --- interior cells i = 1 .. nr-2 ---
    i = np.arange(1, nr - 1)

    flux_diff = (  r_plus[i]  * a_plus[i]   * (T[i+1] - T[i])
                 - r_minus[i] * a_plus[i-1] * (T[i]   - T[i-1])
                ) / (r[i] * dr**2)

    u_f_plus  = 0.5 * (u[i] + u[i+1])
    u_f_minus = 0.5 * (u[i] + u[i-1])

    adv_plus  = r_plus[i]  * np.where(u_f_plus  >= 0, T[i],   T[i+1]) * u_f_plus
    adv_minus = r_minus[i] * np.where(u_f_minus >= 0, T[i-1], T[i])   * u_f_minus

    flux_adv = -gamma[i] * (adv_plus - adv_minus) / (r[i] * dr)

    T_new[i] = T[i] + dt * (flux_diff + flux_adv)

    # --- BCs: Dirichlet inner and outer ---
    T_new[0]  = state.T_inner
    T_new[-1] = state.T_outer

    state.T[:] = T_new