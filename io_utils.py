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

    dim_data = {
        "physics": cfg.get("physics", {}),
        "scaling": cfg.get("scaling", {}),
        "grid":    cfg.get("grid", {}),
        "flow":    cfg.get("flow", {})
    }
    with open(output_dir / "inputs_dimensional.json", "w") as f:
        json.dump(dim_data, f, indent=2)

    dimless_data = {
        "r_0":       state.r_0,
        "r_max":     state.r_max,
        "nr":        state.nr,
        "dr":        state.dr,
        "k_hat":     state.k_hat,
        "D_hat":     state.D_hat,
        "alpha_hat": state.alpha_hat,
        "gamma":     state.gamma,
        "phi":       state.phi,
        "u_inj":     state.u_inj,
        "t_c":       state.t_c,
        "r_c":       state.r_c
    }
    with open(output_dir / "inputs_dimensionless.json", "w") as f:
        json.dump(dimless_data, f, cls=NumpyEncoder, indent=2)


def clear_outputs(base_path):
    base_path = Path(base_path)
    for f in (base_path / "data").glob("outputs_*.h5"):
        f.unlink()
    for f in (base_path / "plots").glob("plot_*.png"):
        f.unlink()


def save_outputs_hdf5(state, base_path, step, t=None, dimensionless=True):
    filename = "outputs_dimensionless.h5" if dimensionless else "outputs_dimensional.h5"
    path = Path(base_path) / "data" / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    s = slice(1, -1)

    if dimensionless:
        data = {
            "T":   state.T,
            "c":   state.c,
            "p":   state.p,
            "u":   state.u,
            "r":   state.grid,
            "phi": state.phi[s],
            "R":   state.R[s],
            "k":   state.k_hat[s],
        }
    else:
        data = {
            "T":   state.T - 273.15,
            "c":   state.c * state.c_c,
            "p":   state.p * state.mu[0] / state.t_c,
            "u":   state.u * state.r_c / state.t_c,
            "r":   state.grid,
            "phi": state.phi[s],
            "R":   state.R[s],
            "k":   state.k_hat[s] * state.r_c**2,
        }

    with h5py.File(path, "a") as f:
        key = f"step_{step}"
        if key in f:
            del f[key]
        grp = f.create_group(key)
        for k, v in data.items():
            grp.create_dataset(k, data=v)
        if t is not None:
            grp.attrs["t"] = t


def plot_1d(state, path, step):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    s      = slice(1, -1)
    r_phys = state.grid * state.r_c
    r_int  = r_phys[s]

    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    axs[0,0].plot(r_phys, state.T - 273.15,                   color="tab:red")
    axs[0,0].set_title("Temperature")
    axs[0,0].set_ylabel(r"$T$ (°C)")

    axs[0,1].plot(r_phys, state.c * state.c_c,                color="tab:blue")
    axs[0,1].set_title("Concentration")
    axs[0,1].set_ylabel(r"$C$ (mol kg$_w^{-1}$)")

    axs[0,2].plot(r_int,  state.R[s],                         color="tab:brown")
    axs[0,2].set_title("Precipitation rate")
    axs[0,2].set_ylabel(r"$S_i$ (mol kg$_w^{-1}$ s$^{-1}$)")

    axs[1,0].plot(r_int,  state.phi[s],                       color="tab:purple")
    axs[1,0].set_title("Porosity")
    axs[1,0].set_ylabel(r"$\phi$ (-)")

    axs[1,1].plot(r_phys, state.k_hat * state.r_c**2,         color="tab:cyan")
    axs[1,1].set_title("Permeability")
    axs[1,1].set_ylabel(r"$k$ (m$^2$)")

    axs[1,2].plot(r_phys, state.p * state.mu[0] / state.t_c,  color="tab:green")
    axs[1,2].set_title("Pressure")
    axs[1,2].set_ylabel(r"$p$ (Pa)")

    for ax in axs.flat:
        ax.set_xlabel(r"$r$ (m)")

    plt.tight_layout()
    plt.savefig(path / f"plot_{step}.png", dpi=150)
    plt.close()
    # plt.rcParams.update(plt.rcParamsDefault)

def _plot_field_over_time(base_path, key, ylabel, name, fname, logy=False):
    """Read the saved snapshots and plot how an interior field evolves in time.

    Left:  field at the well and the domain minimum vs time.
    Right: field(r) profiles at a handful of times (coloured by time).
    """
    h5_path = Path(base_path) / "data" / "outputs_dimensional.h5"
    if not h5_path.exists():
        return

    with h5py.File(h5_path, "r") as f:
        keys  = list(f.keys())
        t     = np.array([f[k].attrs["t"] for k in keys])
        order = np.argsort(t)
        t     = t[order]
        keys  = [keys[i] for i in order]
        field = np.array([f[k][key][:] for k in keys])   # (n_t, nr_interior)
        r     = f[keys[0]]["r"][:]

    r_int = r[1:-1]   # interior fields are saved on slice 1:-1
    out = Path(base_path) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(t, field[:, 0],       color="tab:purple", label=r"at well ($r\approx r_0$)")
    axs[0].plot(t, field.min(axis=1), color="tab:red", ls="--", label="domain minimum")
    axs[0].set_xlabel(r"$t$ (s)")
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(f"{name} over time")
    axs[0].legend()

    idx  = np.linspace(0, len(t) - 1, min(len(t), 6)).astype(int)
    cmap = plt.cm.viridis
    for j, i in enumerate(idx):
        axs[1].plot(r_int, field[i], color=cmap(j / max(len(idx) - 1, 1)),
                    label=f"t={t[i]:.1f}s")
    axs[1].set_xlabel(r"$r$ (m)")
    axs[1].set_ylabel(ylabel)
    axs[1].set_title(f"{name} profile")
    axs[1].legend(fontsize=10)

    if logy:
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(out / fname, dpi=150)
    plt.close()


def plot_porosity_over_time(base_path):
    _plot_field_over_time(base_path, "phi", r"$\phi$ (-)", "Porosity",
                          "porosity_over_time.png")


def plot_permeability_over_time(base_path):
    _plot_field_over_time(base_path, "k", r"$k$ (m$^2$)", "Permeability",
                          "permeability_over_time.png", logy=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)