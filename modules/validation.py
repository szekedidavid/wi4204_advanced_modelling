import numpy as np
from scipy.special import erfc


def solve_analytical_pressure(state):

    r_bar = state.grid
    dr = state.dr
    u_inj_bar = state.u_inj
    k_hat = state.k_hat
    
    f = 1.0 / (r_bar * k_hat)

    segment_integrals = 0.5 * (f[:-1] + f[1:]) * dr
    integral_from_r0 = np.concatenate(([0], np.cumsum(segment_integrals)))
    

    total_integral = integral_from_r0[-1]
    p_analytical = u_inj_bar * (total_integral - integral_from_r0)
    
    return p_analytical

def calculate_errors(numerical, analytical):

    if np.any(np.isnan(numerical)) or np.any(np.isinf(numerical)):
        return {
            "l2": np.nan,
            "linf": np.nan,
            "residual": np.zeros_like(numerical),
            "status": "FAILED: Stabilty Error"
        }

    residual = numerical - analytical
    l2_error = np.sqrt(np.mean(residual**2))
    linf_error = np.max(np.abs(residual))
    
    return {
        "l2": l2_error,
        "linf": linf_error,
        "residual": residual,
        "status": "SUCCESS"
    }

def solve_analytical_transport(state, field_name, t):

    r_bar = state.grid
    r_0 = state.r_0
    
    if field_name == "c":
        D_eff = state.D_hat[0]
        A = state.u_inj * r_0 / state.phi[0]
        front = np.sqrt(r_0**2 + 2.0 * A * t)
        
        val_0 = state.c[-1]
        val_inj = state.bc["c"]["inner"]["value"](0,0,0) if state.bc["c"]["inner"] else 1.0
        
        sol = val_0 + 0.5 * (val_inj - val_0) * erfc((r_bar - front) / np.maximum(np.sqrt(4.0 * D_eff * t), 1e-12))
        return sol

    elif field_name == "T":
        alpha_eff = state.alpha_hat[0]
        gamma = state.gamma[0]
        
        A_T = gamma * state.u_inj * r_0
        front_T = np.sqrt(r_0**2 + 2.0 * A_T * t)
        
        val_0 = state.T[-1]
        val_inj = state.bc["T"]["inner"]["value"](0,0,0) if state.bc["T"]["inner"] else 1.0
        
        sol = val_0 + 0.5 * (val_inj - val_0) * erfc((r_bar - front_T) / np.maximum(np.sqrt(4.0 * alpha_eff * t), 1e-12))
        return sol
    
    else:
        raise ValueError(f"Unknown field: {field_name}")
