import numpy as np

def step_concentration(state, t, dt):
    c = state.c
    u = state.u
    D = state.D

    r = state.grid
    dr = state.dr
    nr = state.nr

    c_new = c.copy()

    for i in range(nr):
        if i == 0:
            bc = state.bc["c"]["inner"]
            if bc["type_bc"] == "dirichlet":
                c_new[i] = bc["value"](r[i], t, state)
            elif bc["type_bc"] == "neumann":
                dcdr = bc["value"](r[i], t, state)
                c_new[i] = c[i+1] - dcdr * dr
            continue

        if i == nr - 1:
            bc = state.bc["c"]["outer"]
            if bc["type_bc"] == "dirichlet":
                c_new[i] = bc["value"](r[i], t, state)
            elif bc["type_bc"] == "neumann":
                dcdr = bc["value"](r[i], t, state)
                c_new[i] = c[i-1] + dcdr * dr
            continue

        d2c = (c[i+1] - 2*c[i] + c[i-1]) / dr**2
        dc = (c[i] - c[i-1]) / dr if u[i] >= 0 else (c[i+1] - c[i]) / dr

        c_new[i] = c[i] + dt * (D[i] * d2c - u[i] * dc)

    state.c[:] = c_new
