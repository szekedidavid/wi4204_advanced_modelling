from pathlib import Path

from solver import Solver
from state import State
from modules import *


def main(sim_id=0):
    base = Path("sims") / str(sim_id)

    modules = {
        "update_k":         update_k,
        "update_transport": update_transport,
        "update_reactions": update_reactions,
        "pressure":         solve_pressure,
        "velocity":         compute_velocity,
        "heat":             step_heat,
        "concentration":    step_concentration,
        "porosity":         step_porosity,
    }

    state = State(
        __import__("json").load(open(base / "config.json"))
    )

    solver = Solver(base, state, modules)
    solver.run()


if __name__ == "__main__":
    main()