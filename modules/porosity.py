import numpy as np


def step_porosity(state, t, dt):
    """Forward Euler step on the porosity ODE.

    dphi/dt = -V_m * R

    where R [mol / kgw / s] is the net precipitation rate (positive = precipitation)
    and V_m [m^3 / mol] is the molar volume of the precipitating species.
    """
    V_m = state.species["molar_volume"]

    dphi_dt = - 1e6 * V_m * state.phi * state.R  # TODO dont forget :3

    state.phi[:] = state.phi + dt * dphi_dt