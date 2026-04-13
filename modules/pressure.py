import numpy as np

def solve_pressure(state, t):
    nr = state.nr
    dr = state.dr
    r = state.grid
    
    A = np.zeros((nr, nr))
    b = np.zeros(nr)
    
    inner_bc = state.bc["p"]["inner"]
    outer_bc = state.bc["p"]["outer"]
    
    # Inner BC
    if inner_bc is not None:
        if inner_bc["type_bc"] == "neumann":
            val = inner_bc["value"]
            g_inner = val(r[0], t, state) if callable(val) else float(val)
            A[0, 0] = 1.0
            A[0, 1] = -1.0
            b[0] = dr * g_inner * (dr / (2.0 * r[0]) - 1.0)
        elif inner_bc["type_bc"] == "dirichlet":
            val = inner_bc["value"]
            p_inner = val(r[0], t, state) if callable(val) else float(val)
            A[0, 0] = 1.0
            b[0] = p_inner
    else:
        A[0,0] = 1.0
        b[0] = 0.0
        
    # Interior nodes
    for i in range(1, nr - 1):
        A[i, i-1] = 1.0 - dr / (2.0 * r[i])
        A[i, i] = -2.0
        A[i, i+1] = 1.0 + dr / (2.0 * r[i])
        b[i] = 0.0
        
    # Outer BC
    if outer_bc is not None:
        if inner_bc is not None and inner_bc["type_bc"] == "neumann" and outer_bc["type_bc"] == "neumann":
            A[-1, -1] = 1.0
            b[-1] = 0.0
        else:
            if outer_bc["type_bc"] == "neumann":
                val = outer_bc["value"]
                g_outer = val(r[-1], t, state) if callable(val) else float(val)
                A[-1, -2] = 1.0
                A[-1, -1] = -1.0
                b[-1] = - dr * g_outer * (1.0 + dr / (2.0 * r[-1]))
            elif outer_bc["type_bc"] == "dirichlet":
                val = outer_bc["value"]
                p_outer = val(r[-1], t, state) if callable(val) else float(val)
                A[-1, -1] = 1.0
                b[-1] = p_outer
    else:
        A[-1,-1] = 1.0
        b[-1] = 0.0
            
    p = np.linalg.solve(A, b)
    state.p[:] = p