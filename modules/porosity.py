import numpy as np


def step_porosity(state, t, dt):
    V_m = np.array([p["molar_volume"] for p in state.precipitates])  # (n_precip,)
    Vr  = V_m @ state.R                                               # (nr,)

    state.phi[1:-1] = state.phi[1:-1] * np.exp(-Vr[1:-1] * dt)