"""
Verification of the coupled pressure analytical solution (eq. A.50).

Takes the last snapshot from sims/1, fits the Gaussian precipitation profile,
then advances only phi -> k -> p with 3 large time steps while freezing T, c, u,
and compares the numerical pressure to the analytical approximation.
"""

import sys
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import exp1

sys.path.insert(0, str(Path(__file__).parents[2]))
from physics.properties import k_of_phi
from modules.pressure import solve_pressure

SIM_DIR  = Path("sims/1")
N_STEPS  = 3
DT_SLOW  = 1e9


def load_last_step(h5path):
    with h5py.File(h5path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        grp  = f[keys[-1]]
        data = {k: grp[k][()] for k in grp.keys()}
        data["t"] = grp.attrs["t"]
    return data


def fit_gaussian(r, R_profile):
    mask = R_profile > 0
    if mask.sum() < 2:
        raise ValueError("Too few nonzero R values to fit.")
    c   = np.polyfit(r[mask]**2, np.log(R_profile[mask]), 1)
    lam = -c[0]
    C0  = np.exp(c[1] + lam * r[mask][0]**2)
    return C0, lam


def analytical_pressure(r, r_max, r_0, mu, u_inj, k_0, V_m, C0, lam, t):
    term1 = (mu * u_inj * r_0 / k_0) * np.log(r_max / r)
    term2 = (3 * mu * u_inj * r_0 * V_m * C0 * np.exp(lam * r_0**2) / (2 * k_0)) \
            * (exp1(lam * r**2) - exp1(lam * r_max**2)) * t
    return term1 + term2


def main():
    cfg   = json.load(open(SIM_DIR / "config.json"))
    snap  = load_last_step(SIM_DIR / "data" / "outputs_dimensional.h5")

    r     = snap["r"]
    r_0, r_max, nr = r[0], r[-1], len(r)
    mu    = cfg["physics"]["mu"]
    k_0   = cfg["physics"]["k_0"]
    phi_0 = cfg["physics"]["phi_0"]
    u_inj = cfg["flow"]["u_inj"]
    t0    = snap["t"]

    from physics.species import load_species
    V_m = load_species(cfg["species"][0]["name"])["molar_volume"]

    R_full       = np.zeros(nr)
    R_full[1:-1] = snap["R"]
    phi          = np.ones(nr) * phi_0
    phi[1:-1]    = snap["phi"]

    C0_fit, lam = fit_gaussian(r[1:-1], snap["R"])
    print(f"Gaussian fit: C0 = {C0_fit:.4e},  lambda = {lam:.4e}")

    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(13, 5))

    p_a_t0 = analytical_pressure(r, r_max, r_0, mu, u_inj, k_0, V_m, C0_fit, lam, t0)
    ax.plot(r, p_a_t0, color="k", lw=1.2, label=r"numerical/analytical at $t_0$")

    times  = []
    errors = []

    for n in range(1, N_STEPS + 1):
        phi  += -V_m * phi * R_full * DT_SLOW
        k_hat = k_of_phi(phi, phi_0, k_0)

        class _S: pass
        st        = _S()
        st.grid   = r
        st.k_hat  = k_hat
        st.mu     = np.ones(nr) * mu
        st.u_inj  = u_inj
        st.r_0    = r_0
        st.p      = np.zeros(nr)
        solve_pressure(st, 0)

        dt_total = n * DT_SLOW
        p_a = analytical_pressure(r, r_max, r_0, mu, u_inj, k_0, V_m, C0_fit, lam, dt_total)

        ax.plot(r, st.p, color=colors[n-1], lw=1.5, label=f"numerical $\\Delta t={dt_total:.0e}$ s")
        ax.plot(r, p_a,  color=colors[n-1], lw=1.0, ls="--", label=f"analytical $\\Delta t={dt_total:.0e}$ s")

        times.append(dt_total)
        errors.append(np.max(np.abs(st.p - p_a)))

    ax.set_xlim(0, 10)
    ax.set_xlabel(r"$r$ (m)")
    ax.set_ylabel(r"$p$ (Pa)")
    ax.set_title("Coupled pressure verification")
    ax.legend(fontsize=9)

    ax_err.plot(times, errors, color="k", marker="o", lw=1.2, zorder=3)
    ax_err.set_xlabel(r"$t$ (s)")
    ax_err.set_ylabel(r"$\max|p_\mathrm{num} - p_\mathrm{ana}|$ (Pa)")
    ax_err.set_title("Max absolute error vs time")
    ax_err.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    plt.tight_layout()
    out = Path("tests/verification/plots")
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "slow_pressure.png", dpi=300)
    print(f"Saved to {out / 'slow_pressure.png'}")
    plt.show()


if __name__ == "__main__":
    main()