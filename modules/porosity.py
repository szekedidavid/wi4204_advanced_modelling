import numpy as np


def step_porosity(state, t, dt):
    """Exact solution of dphi/dt = -phi * sum_i(V_m_i * R_i) with R frozen.

    Since R is frozen over the interval, this is phi(t+dt) = phi(t) * exp(-Vr * dt)
    where Vr = sum_i(V_m_i * R_i).
    """
    V_m = np.array([p["molar_volume"] for p in state.precipitates])  # (n_precip,)
    Vr  = V_m @ state.R                                               # (nr,)

    state.phi[:] = state.phi * np.exp(-Vr * dt)