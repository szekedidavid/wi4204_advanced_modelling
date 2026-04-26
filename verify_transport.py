import numpy as np
import matplotlib.pyplot as plt
from state import State
from io_utils import load_config
from modules.concentration import step_concentration
from modules.heat import step_heat
from modules.validation import solve_analytical_transport
import time

def gaussian_pulse(r, r_center=1.0, width=0.1):
    return np.exp(-((r - r_center)**2) / (2 * width**2))

def verify_field(cfg, field="c", steps_per_frame=10, num_frames=5, dt=0.005):
    cfg["scaling"]["r_c"] = 1.0
    cfg["scaling"]["t_c"] = 1.0
    cfg["scaling"]["c_c"] = 1.0

    cfg["grid"]["nr"] = 200
    cfg["grid"]["r_0"] = 0.1
    cfg["grid"]["r_max"] = 10.0
    
    cfg["flow"]["u_inj"] = 5e-5 
    if field == "c":
        cfg["physics"]["D"] = 0.01
    else:
        cfg["physics"]["alpha"] = 0.01
        
    state = State(cfg)
    
    state.u = state.u_inj * state.r_0 / state.grid
    
    pulse_center = 0.1
    pulse_width = 0.5
    pulse = gaussian_pulse(state.grid, r_center=pulse_center, width=pulse_width)
    if field == "c":
        state.c[:] = pulse
    else:
        state.T[:] = pulse
        
    state.bc = {
        "c": {
            "inner": {"type_bc": "neumann", "value": lambda r, t, s: 0.0},
            "outer": {"type_bc": "neumann", "value": lambda r, t, s: 0.0}
        },
        "T": {
            "inner": {"type_bc": "neumann", "value": lambda r, t, s: 0.0},
            "outer": {"type_bc": "neumann", "value": lambda r, t, s: 0.0}
        }
    }
    
    # Plot Setup
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    r_phys = state.grid * state.r_c
    
    def ic_func(r_val):
        return gaussian_pulse(r_val, r_center=pulse_center, width=pulse_width)

    # Plot initial state (t=0)
    axs[0].plot(r_phys, pulse, ':', color='gray', label="t=0 (Initial)")
    axs[1].axhline(0, color='gray', linestyle=':', alpha=0.5)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'] if field=="c" else ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
    
    current_time = 0.0
    for frame in range(1, num_frames + 1):
        for _ in range(steps_per_frame):
            if field == "c":
                step_concentration(state, 0.0, dt)
            else:
                step_heat(state, 0.0, dt)
        
        current_time += steps_per_frame * dt
        numerical_sol = state.c.copy() if field == "c" else state.T.copy()
        
        # 2. Run Analytical Solution
        label_str = "Concentration" if field == "c" else "Temperature"
        print(f"Evaluating {label_str} Analytical at t={current_time:.2f}...")
        analytical_sol = solve_analytical_transport(state, field, current_time)
        
        c_color = colors[(frame-1) % len(colors)]
        axs[0].plot(r_phys, analytical_sol, '--', color='black', alpha=0.7)
        axs[0].plot(r_phys, numerical_sol, color=c_color, label=f"t={current_time:.2f}")
        
        residual = numerical_sol - analytical_sol
        axs[1].plot(r_phys, residual, color=c_color, label=f"Error t={current_time:.2f}")

    axs[0].set_xlabel("Radius r (m)")
    axs[0].set_xlim(0.0, 3.0)
    axs[0].legend()
    
    axs[1].set_xlabel("Radius r (m)")
    axs[1].set_xlim(0.0, 3.0)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"sims/0/verification_{field}.png")
    print(f"Plot saved to sims/0/verification_{field}.png")

if __name__ == "__main__":
    cfg = load_config("sims/0/config.json")
    verify_field(cfg.copy(), field="c")
    verify_field(cfg.copy(), field="T")
