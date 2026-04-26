import numpy as np
import matplotlib.pyplot as plt
from state import State
from io_utils import load_config
from modules.concentration import step_concentration
from modules.heat import step_heat
from modules.validation import solve_analytical_transport

import time

def run_base_comparison(field="T"):
    print(f"\n--- Comparing {field.upper()} using Base Config ---")
    cfg = load_config("sims/0/config.json")
    
    nr = cfg["grid"].get("nr", 500)
    if nr > 500:
        cfg["grid"]["nr"] = 500
        
    state = State(cfg)
    state.initialize(cfg["ic"])
    state.parse_bc(cfg["bc"])
    
    state.u = state.u_inj * state.r_0 / state.grid
    
    dt = 0.005
    t_target = 0.1
    steps = int(t_target / dt)
        
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    r_phys = state.grid * state.r_c
    
    # Plot Initial State
    initial_sol = state.T.copy() if field == "T" else state.c.copy()
    axs[0].plot(r_phys, initial_sol, ':', color='gray', label="t=0 (Initial)")

    # Run numerics
    for _ in range(steps):
        if field == "c":
            step_concentration(state, 0.0, dt)
        else:
            step_heat(state, 0.0, dt)
            
    numerical_sol = state.c.copy() if field == "c" else state.T.copy()
    
    # Run Analytical Solution
    start_time = time.time()
    analytical_sol = solve_analytical_transport(state, field, t_target)
    
    axs[0].plot(r_phys, analytical_sol, '--', color='black', alpha=0.7, label=f"Analytical t={t_target}")
    axs[0].plot(r_phys, numerical_sol, color='tab:red' if field=="T" else 'tab:blue', label=f"Numerical t={t_target}")
    axs[0].set_xlabel("Radius r (m)")
    axs[0].legend()
    
    residual = numerical_sol - analytical_sol
    l2 = np.sqrt(np.mean(residual**2))
    
    axs[1].plot(r_phys, residual, color='purple')
    axs[1].axhline(0, color='black', linestyle=':', alpha=0.5)
    axs[1].set_title(f"L2 Residual: {l2:.2e}")
    axs[1].set_xlabel("Radius r (m)")
    
    plt.tight_layout()
    plt.savefig(f"sims/0/base_comparison_{field}.png")
    print(f"Saved to sims/0/base_comparison_{field}.png")

if __name__ == "__main__":
    run_base_comparison(field="T")
    run_base_comparison(field="c")
