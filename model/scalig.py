"""
This code computes the M1540 from the apparent magnitude
------------------Inverse Scaling-----------------------

"""
from astropy import units as u
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import trapz
from colors.HZ_colors import fit, f_nu, flux_lambda

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def m_R(filter_fit, lambdas, F_c, const: float, limag):
    """
    Observed magnitude in a filter
    :param filter_fit: filter transmission
    :param lambdas: wavelenghts
    :param F_c: Flux nu
    :param const: hypothetical constant
    :return: observed magnitude in the filter
    """
    if ((trapz(F_c * filter_fit * lambdas, 1 / lambdas) / trapz(filter_fit * const * lambdas, 1 / lambdas)) <= 0):
        m = 99
    else:
        m = -2.5 * np.log10(
            (trapz(F_c * filter_fit * lambdas, 1 / lambdas) / trapz(filter_fit * const * lambdas, 1 / lambdas)))
    if m > limag:
        m = 99
    return m


def scaling(my, z, data, t_filt, const, c):
    """

    Parameters
    ----------
    my: model ref filter magnitude
    z: redhsift
    data: SED
    t_filt: filter transmission
    const: zero point constant
    c: speed of light in A/s

    Returns
    -------

    """
    F_c = fit(data[0] * (1 + z), data[1], t_filt[0])  # f_lambda resampled
    F_nu = f_nu(F_c, t_filt[0], c)  # f_nu resampled
    mm = m_R(t_filt[1], t_filt[0], F_nu, const)
    S = 10 ** (((my - mm) / -2.5))
    f_lam = S * data[1]
    print(S)
    return data[0] * (1 + z), f_lam



