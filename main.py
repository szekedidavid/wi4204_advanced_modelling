from pathlib import Path

from solver import Solver
from state import State
from modules import *


def main(sim_id=0):
    base = Path("sims") / str(sim_id)

    modules = {
        "pressure": solve_pressure,
        "velocity": compute_velocity,
        "heat": step_heat,
        "concentration": step_concentration,
    }

    state = State(
        __import__("json").load(open(base / "config.json"))
    )

    solver = Solver(base, state, modules)
    solver.run()


if __name__ == "__main__":
    main()