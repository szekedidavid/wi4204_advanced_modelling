import numpy as np

def step_heat(state, t, dt):
    T = state.T
    u = state.u
    alpha = state.alpha
    gamma = state.gamma

    r = state.grid
    dr = state.dr
    nr = state.nr

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
        dT = (T[i] - T[i-1]) / dr if u[i] >= 0 else (T[i+1] - T[i]) / dr

        T_new[i] = T[i] + dt * (alpha[i] * d2T - gamma[i] * u[i] * dT)

    state.T[:] = T_new
