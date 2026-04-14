import numpy as np

def step_heat(state, t, dt):
    T = state.T
    u = state.u
    alpha_hat = state.alpha_hat
    gamma = state.gamma

    r = state.grid
    dr = state.dr
    nr = state.nr

    Pe = np.max(np.abs(u) * dr / alpha_hat)
    CFL_adv = np.max(np.abs(u) * dt / dr)
    CFL_diff = np.max(alpha_hat * dt / dr**2)

    if Pe > 2:
        print(f"[T] Peclet warning: Pe = {Pe:.3f} > 2")
    if CFL_adv > 1:
        print(f"[T] CFL advection warning: {CFL_adv:.3f} > 1")
    if CFL_diff > 0.5:
        print(f"[T] CFL diffusion warning: {CFL_diff:.3f} > 0.5")

    T_new = T.copy()

    for i in range(nr):
        if i == 0:
            bc = state.bc["T"]["inner"]
            if bc["type_bc"] == "dirichlet":
                T_new[i] = bc["value"](r[i], t, state)
            elif bc["type_bc"] == "neumann":
                dTdr = bc["value"](r[i], t, state)
                T_new[i] = T[i+1] - dTdr * dr
            continue

        if i == nr - 1:
            bc = state.bc["T"]["outer"]
            if bc["type_bc"] == "dirichlet":
                T_new[i] = bc["value"](r[i], t, state)
            elif bc["type_bc"] == "neumann":
                dTdr = bc["value"](r[i], t, state)
                T_new[i] = T[i-1] + dTdr * dr
            continue

        d2T = (T[i+1] - 2*T[i] + T[i-1]) / dr**2
        dT_diff = (T[i+1] - T[i-1]) / (2*dr)
        dT_adv = (T[i] - T[i-1]) / dr if u[i] >= 0 else (T[i+1] - T[i]) / dr

        laplacian = d2T + (1/r[i]) * dT_diff

        T_new[i] = T[i] + dt * (alpha_hat[i] * laplacian - gamma[i] * u[i] * dT_adv)

    state.T[:] = T_new