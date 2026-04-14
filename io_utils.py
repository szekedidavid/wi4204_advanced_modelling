import json
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def save_state_hdf5(state, path, step, t=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "a") as f:
        key = f"step_{step}"
        if key in f:
            del f[key]
        grp = f.create_group(key)

        for k, v in state.__dict__.items():
            if isinstance(v, np.ndarray):
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