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
    path.parent.mkdir(parents=True, exist_ok=True)

    r = state.grid * state.r_c
    T = state.T * state.dT_scale + state.T_inj
    c = state.c * state.c_c
    p = state.p * state.mu[0] / state.t_c

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    axs[0].plot(r, T, label="T", color="tab:red")
    axs[0].set_ylabel("Temperature ")
    axs[0].legend()
    
    axs[1].plot(r, c, label="c", color="tab:blue")
    axs[1].set_ylabel("Concentration")
    axs[1].legend()
    
    axs[2].plot(r, p, label="p", color="tab:green")
    axs[2].set_ylabel("Pressure (Pa)")
    axs[2].set_xlabel("Radius r (m)")
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(path / f"plot_{step}.png")
    plt.close()