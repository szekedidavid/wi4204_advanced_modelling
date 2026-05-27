"""
Convergence test for the trapezoidal pressure integral in solve_pressure.

With uniform k = k_0, the exact solution is

    p(r) = (mu * u_inj * r_0 / k_0) * ln(R / r)

We solve numerically on grids of increasing nr and measure the L2 error,
expecting second-order convergence from the trapezoid rule.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

MU    = 1e-3
K_0   = 1e-12
U_INJ = 8.8e-3
R_0   = 0.1
R_MAX = 50.0

NR_LIST = [50, 100, 200, 400, 800, 1600]


def numerical_pressure(nr):
    r     = np.linspace(R_0, R_MAX, nr)
    k_hat = np.ones(nr) * K_0
    integrand = 1.0 / (r * k_hat)
    cum = np.zeros(nr)
    for i in range(nr - 1):
        cum[i+1] = cum[i] + 0.5 * (integrand[i] + integrand[i+1]) * (r[i+1] - r[i])
    return r, MU * U_INJ * R_0 * (cum[-1] - cum)


def analytical_pressure(r):
    return (MU * U_INJ * R_0 / K_0) * np.log(R_MAX / r)


errors = []
drs    = []

for nr in NR_LIST:
    r, p_num = numerical_pressure(nr)
    dr       = (R_MAX - R_0) / (nr - 1)
    err      = np.sqrt(np.mean((p_num - analytical_pressure(r))**2))
    errors.append(err)
    drs.append(dr)
    print(f"nr = {nr:5d},  dr = {dr:.4f},  L2 error = {err:.4e}")

drs    = np.array(drs)
errors = np.array(errors)
rates  = np.log(errors[:-1] / errors[1:]) / np.log(drs[:-1] / drs[1:])
print("\nConvergence rates:", np.round(rates, 3))

fig, ax = plt.subplots(figsize=(6, 5))
ax.loglog(drs, errors, "o-", label="numerical error")
ax.loglog(drs, errors[0] * (drs / drs[0])**2, "k--", label=r"$\mathcal{O}(\Delta r^2)$")
ax.set_xlabel(r"$\Delta r$ (m)")
ax.set_ylabel(r"$L^2$ error (Pa)")
ax.set_title("Convergence of trapezoidal pressure integral")
ax.legend()
plt.tight_layout()

out = Path("tests/verification/plots")
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "pressure_convergence.png", dpi=150)
print(f"\nSaved to {out / 'pressure_convergence.png'}")
plt.show()