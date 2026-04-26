import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

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
    
    v_eff = u / phi

    D_dr2 = D_hat / dr**2
    D_2rdr = D_hat / (2.0 * r * dr)
    
    adv_pos = np.maximum(v_eff, 0.0) / dr
    adv_neg = np.minimum(v_eff, 0.0) / dr

    A_low = -dt * (D_dr2 - D_2rdr + adv_pos)
    A_mid = 1.0 + dt * (2.0 * D_dr2 + adv_pos - adv_neg)
    A_up = -dt * (D_dr2 + D_2rdr - adv_neg)
    
    b = c.copy()

    bc_inner = state.bc["c"]["inner"]
    if bc_inner["type_bc"] == "dirichlet":
        A_mid[0] = 1.0
        A_up[0] = 0.0
        b[0] = bc_inner["value"](r[0], t, state)
    elif bc_inner["type_bc"] == "neumann":
        A_mid[0] = -1.0
        A_up[0] = 1.0
        b[0] = bc_inner["value"](r[0], t, state) * dr

    bc_outer = state.bc["c"]["outer"]
    if bc_outer["type_bc"] == "dirichlet":
        A_mid[-1] = 1.0
        A_low[-1] = 0.0
        b[-1] = bc_outer["value"](r[-1], t, state)
    elif bc_outer["type_bc"] == "neumann":
        A_mid[-1] = 1.0
        A_low[-1] = -1.0
        b[-1] = bc_outer["value"](r[-1], t, state) * dr

    A = diags([A_low[1:], A_mid, A_up[:-1]], [-1, 0, 1], format='csr')
    
    c_new = spsolve(A, b)
    state.c[:] = c_new