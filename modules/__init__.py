from .pressure import solve_pressure, compute_velocity
from .heat import step_heat
from .concentration import step_concentration
from .porosity import step_porosity
from physics.properties import k_of_phi, alpha_of_phi, gamma_of_phi
from physics.reactions import reaction_rate


def update_k(state, t):
    k = k_of_phi(state.phi, state.phi_0, state.k_0)
    state.k_hat[:] = k / state.r_c**2


def update_transport(state, t):
    state.alpha_hat[:] = alpha_of_phi(
        state.phi,
        state.c_w, state.rho_w,
        state.c_s, state.rho_s,
        state.lambda_w, state.lambda_s,
        state.t_c, state.r_c
    )
    state.gamma[:] = gamma_of_phi(
        state.phi,
        state.c_w, state.rho_w,
        state.c_s, state.rho_s
    )


def update_reactions(state, t):
    state.R[:] = reaction_rate(
        state.c, state.T, state.phi,
        state.species, state.c_c
    )