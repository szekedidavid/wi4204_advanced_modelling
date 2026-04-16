import numpy as np


def k_of_phi(phi, phi_0, k_0):
    """Kozeny-Carman permeability. Returns dimensional k [m^2]."""
    return k_0 * (phi / phi_0)**3 * ((1 - phi_0) / (1 - phi))**2


def alpha_of_phi(phi, c_w, rho_w, c_s, rho_s, lambda_w, lambda_s, t_c, r_c):
    """Dimensionless thermal diffusivity alpha_hat = alpha * t_c / r_c^2."""
    cw_rhow = c_w * rho_w
    cs_rhos = c_s * rho_s
    denom   = cw_rhow * phi + cs_rhos * (1 - phi)
    alpha   = (lambda_w * phi + lambda_s * (1 - phi)) / denom
    return alpha * t_c / r_c**2


def gamma_of_phi(phi, c_w, rho_w, c_s, rho_s):
    """Dimensionless advection coefficient gamma = (c_w rho_w) / (phi c_w rho_w + (1-phi) c_s rho_s)."""
    cw_rhow = c_w * rho_w
    cs_rhos = c_s * rho_s
    denom   = cw_rhow * phi + cs_rhos * (1 - phi)
    return cw_rhow / denom