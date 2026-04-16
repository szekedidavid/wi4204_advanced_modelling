import numpy as np
from physics.species import load_species

class State:

    def __init__(self, cfg):
        # --- scaling ---
        self.r_c = cfg["grid"]["r_0"]
        self.t_c = cfg["scaling"]["t_c"]
        self.c_c = cfg["scaling"]["c_c"]

        # --- grid ---
        self.r_0   = cfg["grid"]["r_0"]  / self.r_c
        self.r_max = cfg["grid"]["r_max"] / self.r_c
        self.nr    = cfg["grid"]["nr"]

        self.dr   = (self.r_max - self.r_0) / (self.nr - 1)
        self.grid = np.linspace(self.r_0, self.r_max, self.nr)

        # --- temperature BCs (Kelvin) ---
        self.T_inner = cfg["T"]["inner"] + 273.15
        self.T_outer = cfg["T"]["outer"] + 273.15
        T_0          = cfg["T"]["T_0"]   + 273.15

        # --- state variables ---
        self.p = np.zeros(self.nr)
        self.u = np.zeros(self.nr)
        self.T = np.ones(self.nr) * T_0
        self.c = np.ones(self.nr) * cfg["species"][0]["c_0"] / self.c_c  # TODO: (n_species, nr)
        self.R = np.zeros(self.nr)                                         # TODO: (n_species, nr)

        # --- species (single for now) ---
        sp = cfg["species"][0]
        self.c_0   = sp["c_0"]
        self.c_inj = sp["c_inj"]

        # --- porosity ---
        self.phi_0 = cfg["physics"]["phi_0"]
        self.k_0   = cfg["physics"]["k_0"]
        self.phi   = np.ones(self.nr) * self.phi_0

        # --- transport coefficients (owned by physics modules) ---
        self.k_hat     = np.zeros(self.nr)
        self.alpha_hat = np.zeros(self.nr)
        self.gamma     = np.zeros(self.nr)

        # --- diffusivity and viscosity (prescribed, constant) ---
        self.D_hat = np.ones(self.nr) * (cfg["physics"]["D"] * self.t_c / self.r_c**2)
        self.mu    = np.ones(self.nr) * cfg["physics"]["mu"]

        # --- material constants (used by physics/properties) ---
        mat = cfg["material"]
        self.c_w      = mat["c_w"]
        self.rho_w    = mat["rho_w"]
        self.c_s      = mat["c_s"]
        self.rho_s    = mat["rho_s"]
        self.lambda_w = mat["lambda_w"]
        self.lambda_s = mat["lambda_s"]

        # --- flow ---
        self.u_inj = cfg["flow"]["u_inj"] * (self.t_c / self.r_c)

        # --- species parameters (loaded from CSV) ---
        self.species = load_species(cfg["species"][0]["name"])

    def copy(self):
        new = State.__new__(State)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(new, k, v.copy())
            else:
                setattr(new, k, v)
        return new