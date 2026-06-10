import numpy as np
import csv
from pathlib import Path
from physics.species import load_precipitates


PRECIPITATES_DB = Path(__file__).parent / "physics" / "solutes.csv"

_FLOAT_FIELDS = {"molar_volume", "A_arrhenius", "Ea", "SA"}


class State:

    def __init__(self, cfg):
        # --- scaling ---
        self.r_c = 1
        self.t_c = cfg["scaling"]["t_c"]
        self.c_c = cfg["scaling"]["c_c"]

        # --- grid ---
        self.r_0   = cfg["grid"]["r_0"]   / self.r_c
        self.r_max = cfg["grid"]["r_max"]  / self.r_c
        self.nr    = cfg["grid"]["nr"]
        self.dr    = (self.r_max - self.r_0) / (self.nr - 1)
        self.grid  = np.linspace(self.r_0, self.r_max, self.nr)

        # --- temperature BCs (Kelvin) ---
        self.T_inner = cfg["T"]["inner"] + 273.15
        self.T_outer = cfg["T"]["outer"] + 273.15
        T_0          = cfg["T"]["T_0"]   + 273.15

        # --- solutes ---
        solute_cfgs   = cfg["solutes"]                         # list of {name, D, c_0, c_inj}
        self.n_solutes     = len(solute_cfgs)
        self.solute_names  = [s["name"] for s in solute_cfgs]
        self.solute_index  = {s: i for i, s in enumerate(self.solute_names)}
        self.D_hat    = np.array([
            s["D"] * self.t_c / self.r_c**2 for s in solute_cfgs
        ])                                                     # (n_solutes,)
        self.c_0      = np.array([s["c_0"]   for s in solute_cfgs])
        self.c_inj    = np.array([s["c_inj"] for s in solute_cfgs])
        self.c        = np.outer(self.c_0 / self.c_c, np.ones(self.nr))  # (n_solutes, nr)

        # --- precipitates ---
        self.precipitates = load_precipitates(cfg["precipitates"], self.solute_names)
        self.n_precip = len(self.precipitates)

        # nu matrix: (n_precip, n_solutes)
        self.nu = np.array([
            [p["nu"].get(s, 0) for s in self.solute_names]
            for p in self.precipitates
        ], dtype=float)

        # --- reaction rates ---
        self.R = np.zeros((self.n_precip, self.nr))            # (n_precip, nr)

        # --- state variables ---
        self.p = np.zeros(self.nr)
        self.u = np.zeros(self.nr)
        self.T = np.ones(self.nr) * T_0

        # --- porosity ---
        self.phi_0 = cfg["physics"]["phi_0"]
        self.k_0   = cfg["physics"]["k_0"]
        self.phi   = np.ones(self.nr) * self.phi_0

        # --- transport coefficients ---
        self.k_hat     = np.zeros(self.nr)
        self.alpha_hat = np.zeros(self.nr)
        self.gamma     = np.zeros(self.nr)

        # --- viscosity ---
        self.mu = np.ones(self.nr) * cfg["physics"]["mu"]

        # --- material constants ---
        mat           = cfg["material"]
        self.c_w      = mat["c_w"]
        self.rho_w    = mat["rho_w"]
        self.c_s      = mat["c_s"]
        self.rho_s    = mat["rho_s"]
        self.lambda_w = mat["lambda_w"]
        self.lambda_s = mat["lambda_s"]

        # --- flow ---
        self.u_inj = cfg["flow"]["u_inj"] * (self.t_c / self.r_c)

    def copy(self):
        new = State.__new__(State)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(new, k, v.copy())
            else:
                setattr(new, k, v)
        return new