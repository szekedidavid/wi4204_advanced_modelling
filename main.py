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

    # --- Post-Processing: Analytical Validation ---
    print("\nRunning Analytical Validation...")
    from modules import solve_analytical_pressure
    import io_utils as io
    
    p_analytical = solve_analytical_pressure(state)
    io.plot_comparison(state, p_analytical, "p", base / "validation_pressure.png")
    print(f"Validation plot saved to: {base / 'validation_pressure.png'}")


if __name__ == "__main__":
    main()