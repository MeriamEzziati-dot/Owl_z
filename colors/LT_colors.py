import numpy as np
from colors.HZ_colors import f_nu, fit, m_R
from scipy.integrate import trapz

def m_Rr(filter_fit, lambdas, F_c, const: float):
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
    return m


def colors_lt(data, const, filt,c):
    template = data[1]
    lambdas = data[0]
    L = filt[0]
    T_inter = fit(lambdas, template, L)
    m = m_Rr(filt[1], L, f_nu(T_inter, L,c), const)

    return m
