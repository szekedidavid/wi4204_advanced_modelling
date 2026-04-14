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

    r = state.grid

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].plot(r, state.T)
    axs[0].set_title("T")

    axs[1].plot(r, state.c)
    axs[1].set_title("c")

    axs[2].plot(r, state.p)
    axs[2].set_title("p")

    axs[3].plot(r, state.u)
    axs[3].set_title("u")

    for ax in axs:
        ax.set_xlabel("r")

    plt.tight_layout()
    plt.savefig(path / f"plot_{step}.png")
    plt.close()

def animate_live(state, step):
    import matplotlib.pyplot as plt

    if not hasattr(animate_live, "_fig"):
        plt.ion()
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        animate_live._fig = fig
        animate_live._axs = axs
        animate_live._lines = [ax.plot(state.grid, np.zeros_like(state.grid))[0] for ax in axs]
        for ax, title in zip(axs, ["T", "c", "p", "u"]):
            ax.set_xlabel("r")
            ax.set_title(title)

    lines = animate_live._lines
    axs = animate_live._axs
    fields = [state.T, state.c, state.p, state.u]

    for line, ax, field in zip(lines, axs, fields):
        line.set_xdata(state.grid)
        line.set_ydata(field)
        ax.relim()
        ax.autoscale_view()

    animate_live._fig.suptitle(f"step {step}")
    animate_live._fig.canvas.draw()
    animate_live._fig.canvas.flush_events()