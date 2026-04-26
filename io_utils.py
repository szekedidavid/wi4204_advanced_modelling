import json
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def save_inputs_json(state, cfg, base_path):
    output_dir = Path(base_path) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dimensional Inputs
    dim_data = {
        "physics": cfg.get("physics", {}),
        "scaling": cfg.get("scaling", {}),
        "grid": cfg.get("grid", {}),
        "flow": cfg.get("flow", {})
    }
    with open(output_dir / "inputs_dimensional.json", "w") as f:
        json.dump(dim_data, f, indent=2)
        
    # 2. Dimensionless Inputs
    dimless_data = {
        "r_0": state.r_0,
        "r_max": state.r_max,
        "nr": state.nr,
        "dr": state.dr,
        "k_hat": state.k_hat,
        "D_hat": state.D_hat,
        "alpha_hat": state.alpha_hat,
        "gamma": state.gamma,
        "phi": state.phi,
        "u_inj": state.u_inj,
        "t_c": state.t_c,
        "r_c": state.r_c
    }
    with open(output_dir / "inputs_dimensionless.json", "w") as f:
        json.dump(dimless_data, f, cls=NumpyEncoder, indent=2)

def save_outputs_hdf5(state, base_path, step, t=None, dimensionless=True):
    filename = "outputs_dimensionless.h5" if dimensionless else "outputs_dimensional.h5"
    path = Path(base_path) / "data" / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    if dimensionless:
        data_to_save = {
            "T": state.T,
            "c": state.c,
            "p": state.p,
            "u": state.u,
            "r": state.grid
        }
    else:
        # Rescale to physical units
        data_to_save = {
            "T": state.T * state.dT_scale + state.T_inj,
            "c": state.c * state.c_c,
            "p": state.p * state.mu[0] / state.t_c,
            "u": state.u * state.r_c / state.t_c,
            "r": state.grid * state.r_c
        }

    with h5py.File(path, "a") as f:
        key = f"step_{step}"
        if key in f:
            del f[key]
        grp = f.create_group(key)
        for k, v in data_to_save.items():
            grp.create_dataset(k, data=v)
        if t is not None:
            grp.attrs["t"] = t


def plot_1d(state, path, step):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    r_phys = state.grid * state.r_c
    T_phys = state.T * state.dT_scale + state.T_inj
    c_phys = state.c * state.c_c
    p_phys = state.p * state.mu[0] / state.t_c
    u_phys = state.u * state.r_c / state.t_c

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].plot(r_phys, T_phys, color="tab:red")
    axs[0].set_title("Temperature (°C)")
    axs[0].set_ylabel("T")

    axs[1].plot(r_phys, c_phys, color="tab:blue")
    axs[1].set_title("Concentration")
    axs[1].set_ylabel("c")

    axs[2].plot(r_phys, p_phys, color="tab:green")
    axs[2].set_title("Pressure (Pa)")
    axs[2].set_ylabel("p")

    axs[3].plot(r_phys, u_phys, color="tab:orange")
    axs[3].set_title("Velocity (m/s)")
    axs[3].set_ylabel("u")

    for ax in axs:
        ax.set_xlabel("Radius r (m)")

    plt.tight_layout()
    plt.savefig(path / f"plot_{step}.png")
    plt.close()

def animate_live(state, step):
    if not hasattr(animate_live, "_fig"):
        plt.ion()
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        animate_live._fig = fig
        animate_live._axs = axs
        animate_live._lines = [ax.plot(state.grid * state.r_c, np.zeros_like(state.grid))[0] for ax in axs]
        titles = ["Temp (°C)", "Conc", "Pres (Pa)", "Vel (m/s)"]
        for ax, title in zip(axs, titles):
            ax.set_xlabel("r (m)")
            ax.set_title(title)

    lines = animate_live._lines
    axs = animate_live._axs
    
    T_phys = state.T * state.dT_scale + state.T_inj
    c_phys = state.c * state.c_c
    p_phys = state.p * state.mu[0] / state.t_c
    u_phys = state.u * state.r_c / state.t_c
    
    fields = [T_phys, c_phys, p_phys, u_phys]

    for line, ax, field in zip(lines, axs, fields):
        line.set_xdata(state.grid * state.r_c)
        line.set_ydata(field)
        ax.relim()
        ax.autoscale_view()

    animate_live._fig.suptitle(f"step {step}")
    animate_live._fig.canvas.draw()
    animate_live._fig.canvas.flush_events()

def plot_comparison(state, analytical, field_name, path):
    """
    Plots a dashboard comparing numerical and analytical solutions in physical space.
    """
    from modules.validation import calculate_errors
    
    numerical_dimless = getattr(state, field_name)
    analytical_dimless = analytical
    r_phys = state.grid * state.r_c
    
    # Rescale for plot
    if field_name == "p":
        numerical = numerical_dimless * state.mu[0] / state.t_c
        analytical = analytical_dimless * state.mu[0] / state.t_c
        unit = "Pa"
    else:
        numerical = numerical_dimless
        analytical = analytical_dimless
        unit = "dimless"
    
    errors = calculate_errors(numerical, analytical)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    axs[0].plot(r_phys, analytical, '--', label="Analytical (Exact)", color="black", alpha=0.7)
    axs[0].plot(r_phys, numerical, label="Numerical (Solver)", color="tab:red", alpha=0.8)
    axs[0].set_xlabel("Radius r (m)")
    axs[0].set_ylabel(f"Value ({unit})")
    axs[0].legend()
    
    axs[1].plot(r_phys, errors["residual"], color="tab:purple")
    axs[1].axhline(0, color='black', linestyle=':', alpha=0.5)
    axs[1].set_title(f"L2 Error: {errors['l2']:.2e}")
    axs[1].set_xlabel("Radius r (m)")
    axs[1].set_ylabel(f"Error ({unit})")
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_evolution(state, base_path, steps=[0, -1]):
    """
    Creates a plot comparing specific timesteps in PHYSICAL units.
    """
    path = Path(base_path) / "data" / "outputs_dimensionless.h5"
    if not path.exists():
        return
        
    with h5py.File(path, "r") as f:
        available_steps = sorted(list(f.keys()), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        
        target_keys = []
        for s in steps:
            key = f"step_{s}"
            if key in available_steps:
                target_keys.append(key)
            elif isinstance(s, int) and s < 0 and abs(s) <= len(available_steps):
                target_keys.append(available_steps[s])
        
        if not target_keys:
            return

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        linestyles = ['--', '-', ':']

        for i, key in enumerate(target_keys):
            data = f[key]
            r_bar = data["r"][:]
            T_bar = data["T"][:]
            c_bar = data["c"][:]
            
            # Rescale to physical
            r_phys = r_bar * state.r_c
            T_phys = T_bar * state.dT_scale + state.T_inj
            c_phys = c_bar * state.c_c
            
            t_dimless = f[key].attrs.get('t', 0.0)
            t_phys = t_dimless * state.t_c
            label = f"t = {t_phys:.0f}s"
            
            axs[0].plot(r_phys, T_phys, label=label, alpha=0.5 + 0.5*(i/len(target_keys)), linestyle=linestyles[i % 3], color="tab:red")
            axs[1].plot(r_phys, c_phys, label=label, alpha=0.5 + 0.5*(i/len(target_keys)), linestyle=linestyles[i % 3], color="tab:blue")

        axs[0].set_title("Temperature Evolution")
        axs[0].set_ylabel("Temperature (°C)")
        
        axs[1].set_title("Concentration Evolution")
        axs[1].set_ylabel("Concentration")
        
        for ax in axs:
            ax.set_xlabel("Radius r (m)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(base_path) / "front_evolution.png")
        plt.close()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
