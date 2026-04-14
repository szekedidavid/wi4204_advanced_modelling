import numpy as np

def solve_analytical_pressure(state):

    r_bar = state.grid
    R_bar = state.r_max
    u_inj_bar = state.u_inj
    k_hat = np.mean(state.k_hat)
    
    p_analytical = (u_inj_bar / k_hat) * np.log(R_bar / r_bar)
    
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
