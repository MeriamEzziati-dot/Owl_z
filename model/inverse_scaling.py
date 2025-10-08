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

def fit(x, y, z):
    """
    Resampling function
    :param x: initial abscissa
    :param y: initial function
    :param z:  abscissa to resample with
    :return: function resampled by z
    """
    fit_y = interp1d(x, y, bounds_error=False, fill_value=0)
    Zy = fit_y(z)
    return Zy
def Flux_density(filter_fit, lambdas, F_c):
    """
    :param filter_fit: Transmission filter in photon interpolated
    :param lambdas: wavelenghts
    :param F_c: flux nu interpolated
    :return: flux density in the corresponding filter
    """

    return simps(F_c * filter_fit * lambdas, 1 / lambdas) / simps(filter_fit * lambdas, 1 / lambdas)




def m_R(filter_fit, lambdas, F_c, const: float):
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
def m_f_nu(fnu):
    return -2.5 * np.log10(fnu) - 48.6
def newtemplate(my, z, data,lambdas, xJ,J, const, c):
    F_c = interp1d(lambdas* (1 + z), data, bounds_error=False, fill_value=0)(xJ)
    F_nu = F_c* (xJ ** 2) / c
    mm = m_R(J, xJ, F_nu, const)
    S = 10 ** (((my - mm) / -2.5))
    f_lam= S* data
    return lambdas*(1+z),f_lam

def m_1450(z, m,logdl):
    """
    :param zr: float
    redshift
    :param m1450: float
    absolute magnitude at 1450
    :return:m
     the observed magnitude at 1450(1+z) Angstroms
    """
    return m + 5 - 5 * logdl + 2.5 * np.log10(
        (z + 1))

def m_1450_ob(my, redshift, data, Hf, const, c,logdl,xx):

    fluxes=data[1]
    lambdas=data[0]
    Tempz=fit(newtemplate(my, redshift, fluxes,lambdas, xx,Hf, const, c)[0], newtemplate(my, redshift, fluxes,lambdas,xx, Hf, const, c)[1], xx)
    Fh = Flux_density(Hf, xx, f_nu(Tempz, xx, c))
    C = my + 2.5 * np.log10(Fh)

    F1450= Tempz[(xx > 1450.*(1+redshift)-10) & (xx < 1450.*(1+redshift)+10)]
    L1450=xx[(xx > 1450.*(1+redshift)-10) & (xx < 1450.*(1+redshift)+10)]
    T1450=np.ones(len(F1450))

    Fnu1450=Flux_density(T1450, L1450, f_nu(F1450, L1450, c))
    m1450=-2.5 * np.log10(Fnu1450) + C

    M1450=m1450 - 5*logdl+ 5 + 2.5*np.log10(1+redshift)


    return M1450


def m_1450_ob_G_old(my, z, data, J, const, c,logdl):
    F_c = fit(data[0] * (1 + z), data[1], J[0])  # f_lambda resampled
    F_nu = f_nu(F_c, J[0], c)  # f_nu resampled
    mm = m_R(J[1], J[0], F_nu, const)
    S = 10 ** (((my - mm) / -2.5))
    f_lam= S* data[1][(data[0]>4480)&(data[0]<4520)].mean(0)
    f_nuu = f_nu(f_lam, 4500 * (1 + z), c)
    mobs = m_f_nu(f_nuu)
    return mobs - 5 * logdl+ 2.5 * np.log10(1 + z) + 5



def m_1450_ob_G(my, redshift, data, Hf, const, c,logdl,xx):

    fluxes=data[1]
    lambdas=data[0]
    Tempz=fit(newtemplate(my, redshift, fluxes,lambdas, xx,Hf, const, c)[0], newtemplate(my, redshift, fluxes,lambdas,xx, Hf, const, c)[1], xx)
    Fh = Flux_density(Hf, xx, f_nu(Tempz, xx, c))
    C = my + 2.5 * np.log10(Fh)

    F1450= Tempz[(xx > 4500.*(1+redshift)-10) & (xx < 4500.*(1+redshift)+10)]
    L1450=xx[(xx > 4500.*(1+redshift)-10) & (xx < 4500.*(1+redshift)+10)]
    T1450=np.ones(len(F1450))

    Fnu1450=Flux_density(T1450, L1450, f_nu(F1450, L1450, c))
    m1450=-2.5 * np.log10(Fnu1450) + C

    M1450=m1450 - 5*logdl+ 5 + 2.5*np.log10(1+redshift)


    return M1450
