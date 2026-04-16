import numpy as np

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
