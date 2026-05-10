import numpy as np


def step_heat(state, t, dt):
    T         = state.T
    u         = state.u
    alpha_hat = state.alpha_hat
    gamma     = state.gamma
    r         = state.grid
    dr        = state.dr
    nr        = state.nr

    CFL_adv  = np.max(np.abs(gamma * u) * dt / dr)
    CFL_diff = np.max(alpha_hat * dt / dr**2)

    if CFL_adv > 1:
        print(f"[T] CFL advection warning: {CFL_adv:.3f} > 1")
    if CFL_diff > 0.5:
        print(f"[T] CFL diffusion warning: {CFL_diff:.3f} > 0.5")

    r_plus  = r + 0.5 * dr
    r_minus = r - 0.5 * dr

    a_face = 2 * alpha_hat[:-1] * alpha_hat[1:] / (alpha_hat[:-1] + alpha_hat[1:])

    i = np.arange(1, nr - 1)

    flux_diff = (  r_plus[i]  * a_face[i]   * (T[i+1] - T[i])
                 - r_minus[i] * a_face[i-1] * (T[i]   - T[i-1])
                ) / (r[i] * dr**2)

    flux_adv = gamma[i] * u[i] * (T[i] - T[i-1]) / dr

    T_new = T.copy()
    T_new[i] = T[i] + dt * (flux_diff - flux_adv)

    T_new[0]  = state.T_inner
    T_new[-1] = state.T_outer

    state.T[:] = T_new