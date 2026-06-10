from pathlib import Path
import io_utils as io


class Solver:
    def __init__(self, sim_path, state, modules):
        self.base    = Path(sim_path)
        self.state   = state
        self.modules = modules

        self.cfg = io.load_config(self.base / "config.json")

        self.t_c           = self.cfg["scaling"]["t_c"]
        self.dt_fast       = self.cfg["time"]["dt_fast"]      / self.t_c
        self.t_fast_init   = self.cfg["time"]["t_fast_init"]  / self.t_c
        self.t_fast        = self.cfg["time"]["t_fast"]       / self.t_c
        self.dt_slow       = self.cfg["time"]["dt_slow"]      / self.t_c
        self.t_slow        = self.cfg["time"]["t_slow"]       / self.t_c
        self.save_interval = self.cfg["time"]["save_interval"]

        self.n_fast_init = int(self.t_fast_init / self.dt_fast)
        self.n_fast      = int(self.t_fast      / self.dt_fast)
        self.n_slow      = int(self.t_slow      / self.dt_slow)

        self.modules["update_k"](self.state, 0)
        self.modules["update_transport"](self.state, 0)
        self.modules["pressure"](self.state, 0)
        self.modules["velocity"](self.state, 0)

        io.save_inputs_json(self.state, self.cfg, self.base)

    def _fast_block(self, t_outer, n_steps):
        for inner in range(n_steps):
            t = t_outer + inner * self.dt_fast
            self.modules["update_reactions"](self.state, t)
            self.modules["heat"](self.state, t, self.dt_fast)
            self.modules["concentration"](self.state, t, self.dt_fast)
            if inner % 100 == 0:
                print(f"  [fast {inner}/{n_steps}]  t = {t * self.t_c:.2e} s")

    def _slow_update(self, t_outer):
        self.modules["porosity"](self.state, t_outer, self.dt_slow)
        self.modules["update_k"](self.state, t_outer)
        self.modules["update_transport"](self.state, t_outer)
        self.modules["pressure"](self.state, t_outer)
        self.modules["velocity"](self.state, t_outer)

    def run(self):
        io.clear_outputs(self.base)

        # step 0: initial condition
        io.save_outputs_hdf5(self.state, self.base, 0, 0.0, dimensionless=True)
        io.save_outputs_hdf5(self.state, self.base, 0, 0.0, dimensionless=False)
        io.plot_1d(self.state, self.base / "plots", 0)
        print("step 0: initial condition saved")

        # step 1: initial equilibration + first porosity update
        print(f"[init]  {self.n_fast_init} fast steps for initial equilibration")
        self._fast_block(0.0, self.n_fast_init)
        self._slow_update(0.0)
        io.save_outputs_hdf5(self.state, self.base, 1, 0.0, dimensionless=True)
        io.save_outputs_hdf5(self.state, self.base, 1, 0.0, dimensionless=False)
        io.plot_1d(self.state, self.base / "plots", 1)
        print("step 1: initial equilibration saved")

        # subsequent outer steps
        for outer in range(self.n_slow):
            t_outer = (outer + 1) * self.dt_slow
            print(f"[slow {outer + 1}/{self.n_slow}]  t = {t_outer * self.t_c:.2e} s")

            self._fast_block(t_outer, self.n_fast)
            self._slow_update(t_outer)

            if outer % self.save_interval == 0:
                io.save_outputs_hdf5(self.state, self.base, outer + 2, t_outer, dimensionless=True)
                io.save_outputs_hdf5(self.state, self.base, outer + 2, t_outer, dimensionless=False)
                io.plot_1d(self.state, self.base / "plots", outer + 2)

        print("done")