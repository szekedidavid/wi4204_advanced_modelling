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

    r = state.grid

    plt.figure()
    plt.plot(r, state.T, label="T")
    plt.plot(r, state.c, label="c")
    plt.plot(r, state.p, label="p")
    plt.legend()
    plt.xlabel("r")
    plt.savefig(path / f"plot_{step}.png")
    plt.close()