import numpy as np

def step_concentration(state, t, dt):
    c = state.c
    u = state.u
    D_hat = state.D_hat
    phi = state.phi
    
    t_c = state.t_c
    c_c = state.c_c

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

        R = 0.0
        
        c_new[i] = c[i] + dt * (D_hat[i] * d2c - (1.0 / phi[i]) * u[i] * dc - (t_c / c_c) * R)

    state.c[:] = c_new
