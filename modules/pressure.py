import numpy as np

from .porosity import porosity_rate


def _cumtrapz(y, x):
    """Cumulative trapezoidal integral of y over x, starting at 0."""
    cum = np.zeros_like(y)
    for i in range(len(x) - 1):
        cum[i+1] = cum[i] + 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    return cum


def radial_flux(state):
    """Storage-corrected radial Darcy flux Q(r) = r u(r), from the water mass balance.

    Incompressible water with the porosity change from clogging as pore storage:

        dphi/dt + (1/r) d(r u)/dr = 0.

    Integrating once in 1D radial (injection r_0 u_inj at the well minus the
    cumulative pore storage) gives

        Q(r) = r_0 u_inj - integral_{r_0}^{r} r' (dphi/dt) dr'.

    Derived entirely from the current state (porosity_rate); reduces to the
    constant flux r_0 u_inj in the uncoupled limit dphi/dt -> 0.
    """
    r = state.grid
    return state.r_0 * state.u_inj - _cumtrapz(r * porosity_rate(state), r)


def solve_pressure(state, t):
    """Quasi-steady pressure from the radial flux via Darcy's law.

    With Q(r) from radial_flux and u = -(k/mu) dp/dr, integrating inward from
    p(r_max) = 0 gives

        p(r) = mu integral_{r}^{r_max} Q / (r' k_hat) dr'.

    Reduces exactly to the previous analytic pressure when dphi/dt -> 0.
    """
    r     = state.grid
    k_hat = state.k_hat
    mu    = state.mu[0]

    integrand = radial_flux(state) / (r * k_hat)
    cum = _cumtrapz(integrand, r)
    state.p[:] = mu * (cum[-1] - cum)


def compute_velocity(state, t):
    """Darcy velocity from the radial flux: u(r) = Q(r) / r = -(k/mu) dp/dr.

    Recomputed from the current state (same mass balance that sets the pressure),
    so u is consistent with the pressure field, satisfies u(r_0) = u_inj exactly,
    and reduces to u_inj r_0 / r in the uncoupled limit.
    """
    state.u[:] = radial_flux(state) / state.grid