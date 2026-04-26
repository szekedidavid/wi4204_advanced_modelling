import numpy as np
from state import State
from io_utils import load_config
from modules.validation import solve_analytical_transport

cfg = load_config("sims/0/config.json")
state = State(cfg)
state.initialize(cfg["ic"])
state.parse_bc(cfg["bc"])

ana_c = solve_analytical_transport(state, "c", 0.02)
print("Min C:", np.min(ana_c))
print("Max C:", np.max(ana_c))

ana_T = solve_analytical_transport(state, "T", 0.02)
print("Min T:", np.min(ana_T))
print("Max T:", np.max(ana_T))
