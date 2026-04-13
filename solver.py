# solver.py

from pathlib import Path
import io_utils as io


class Solver:
    def __init__(self, sim_path, state, modules):
        self.base = Path(sim_path)
        self.state = state
        self.modules = modules

        self.cfg = io.load_config(self.base / "config.json")

        self.dt = self.cfg["time"]["dt"]
        self.t_end = self.cfg["time"]["t_end"]
        self.save_interval = self.cfg["time"]["save_interval"]

        self.n_steps = int(self.t_end / self.dt)

        self.state.initialize(self.cfg["ic"])
        self.state.parse_bc(self.cfg["bc"])

    def step(self, t):
        self.modules["pressure"](self.state, t)
        self.modules["velocity"](self.state, t)
        self.modules["heat"](self.state, t, self.dt)
        self.modules["concentration"](self.state, t, self.dt)

    def run(self):
        for n in range(self.n_steps):
            t = n * self.dt

            self.step(t)

            if n % self.save_interval == 0:
                io.save_state_hdf5(self.state, self.base / "data" / "state.h5", n)
                io.plot_1d(self.state, self.base / "plots", n)

                print(f"step {n}/{self.n_steps}")