import numpy as np

FACTOR = 1

def porosity_rate(state):
    """Porosity rate of change dphi/dt = -V_m * phi * rho_w * R.

    R [mol / kgw / s] is the net precipitation rate (positive = precipitation)
    and V_m [m^3 / mol] is the molar volume of the precipitating species, so
    precipitation (R > 0) reduces porosity (clogging).

    Single source of truth: used both by the porosity step below and by the
    pore-storage term in the pressure mass balance (modules/pressure.py).
    """
    V_m = state.species["molar_volume"]
    return - FACTOR * V_m * state.phi * state.rho_w * state.R


def step_porosity(state, t, dt):
    """Forward Euler step on the porosity ODE, dphi/dt = -V_m * phi * rho_w * R."""
    dphi_dt = porosity_rate(state)

    # Stability of the explicit step: phi_new = phi*(1 - dt*K), K = |dphi/dt|/phi.
    # dt*K >= 1 lets porosity overshoot to <= 0 (then k, R blow up -> overflow).
    phi = state.phi
    K = np.max(np.abs(dphi_dt[phi > 0]) / phi[phi > 0]) if np.any(phi > 0) else 0.0
    if dt * K > 1:
        print(f"[phi] clogging-rate warning: dt*K = {dt * K:.3f} > 1 "
              f"(porosity may go negative; reduce dt or the reaction rate)")

    state.phi[:] = phi + dt * dphi_dt