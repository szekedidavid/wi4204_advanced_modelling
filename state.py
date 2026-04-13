import numpy as np

from bc import p_inner

class State:
    def __init__(self, cfg):
        self.r_0 = cfg["grid"]["r_0"]
        self.r_max = cfg["grid"]["r_max"]
        self.nr = cfg["grid"]["nr"]

        self.dr = (self.r_max - self.r_0) / self.nr
        self.grid = np.linspace(self.r_0, self.r_max, self.nr)

        self.p = np.zeros(self.nr)
        self.u = np.zeros(self.nr)

        self.T = np.zeros(self.nr)
        self.c = np.zeros(self.nr)

        self.phi = np.ones(self.nr) * cfg["physics"]["phi"]
        self.k = np.ones(self.nr) * cfg["physics"]["k"]
        self.D = np.ones(self.nr) * cfg["physics"]["D"]
        self.alpha = np.ones(self.nr) * cfg["physics"]["alpha"]
        self.gamma = np.ones(self.nr) * cfg["physics"]["gamma"]
        self.mu = np.ones(self.nr) * cfg["physics"]["mu"]

        self.u_inj = cfg["flow"]["u_inj"]

        self.bc = {}

    def initialize(self, ic_cfg):
        def evaluate(spec):
            if isinstance(spec, (int, float)):
                return np.ones(self.nr) * spec

            if isinstance(spec, dict):
                if spec["type"] == "constant":
                    return np.ones(self.nr) * spec["value"]

                if spec["type"] == "function":
                    func = eval(spec["expr"])
                    return func(self.grid)

            raise ValueError("Invalid IC specification")

        if "T" in ic_cfg:
            self.T[:] = evaluate(ic_cfg["T"])
        if "c" in ic_cfg:
            self.c[:] = evaluate(ic_cfg["c"])
        if "p" in ic_cfg:
            self.p[:] = evaluate(ic_cfg["p"])
        if "u" in ic_cfg:
            self.u[:] = evaluate(ic_cfg["u"])

    def parse_bc(self, bc_cfg):
        def make(spec):
            if spec is None:
                return None

            if spec["type"] == "constant":
                val = spec["value"]
                return lambda r, t, state: val

            if spec["type"] == "function":
                func = eval(spec["value"])
                return func

            raise ValueError("Invalid BC specification")

        self.bc = {}

        for field, sides in bc_cfg.items():
            self.bc[field] = {}
            for side, spec in sides.items():
                if spec is None:
                    self.bc[field][side] = None
                else:
                    self.bc[field][side] = {
                        "type_bc": spec["type_bc"],
                        "value": make(spec)
                    }

    def copy(self):
            new = State.__new__(State)
            for k, v in self.__dict__.items():
                if isinstance(v, np.ndarray):
                    setattr(new, k, v.copy())
                else:
                    setattr(new, k, v)
            return new
