"""
-----------------------------------------------------
Author : Meriam Ezziati
Date   : 8 April 2022
------------------------------------------------------

MODULE QSO: Functions to calculate the marginal distribution for the bayesian
selection method for high redshift quasars

"""
import warnings

import numpy as np
from scipy import special, interpolate
from astropy.cosmology import FlatLambdaCDM
from decimal import Decimal
from model.inverse_scaling import *
import model.high_z_gal as gal
warnings.filterwarnings("ignore")

cosmo = FlatLambdaCDM(H0=67.3, Om0=0.315)

def m_1450_ob_g(my, z, data, J, const, c,logdl):
    F_c = fit(data[0] * (1 + z), data[1], J[0])  # f_lambda resampled
    F_nu = f_nu(F_c, J[0], c)  # f_nu resampled
    mm = m_R(J[1], J[0], F_nu, const)
    S = 10 ** (((my - mm) / -2.5))
    #f_lam = S * f_template(data)
    f_lam= S* data[1][(data[0]>1490)&(data[0]<1510)].mean(0)
    f_nuu = f_nu(f_lam, 1450 * (1 + z), c)
    mobs = m_f_nu(f_nuu)
    return mobs - 5 * logdl+ 2.5 * np.log10(1 + z) + 5
def completude(m):
    """
    Completeness for the four fields fit done by Sarah using Sexstractor
    :param x: magnitude Y
    :param chp: field W1,W2,W3 or W4
    :return: Completeness
    """
    return 1.0


def luminosity_function(x, z):
    """ Computes the luminosity function for a quasar z>6) using a LF for a quasar at z=6
    z: redshift of desired quasar
    x:  magnitude
    k: evolution parameter pour SDSS et CFHQS
    a: the faint end slope  pour SDSS et CFHQS
    b: the bright end slope pour SDSS et CFHQS
    Mstar:break magnitude
    LFstar:normalisation
    """
    k = -0.47  # Barnett 2019
    a = -1.5  # Barnett 2019
    b = -2.81  # Barnett 2019
    Mstar = -25.13
    LFstar = 1.14 * 1e-8
    return (LFstar * pow(10, k * (z - 6.))) / (
            pow(10, 0.4 * (a + 1) * (x - Mstar)) + pow(10, 0.4 * (b + 1) * (x - Mstar)))


def M_abs(m, z,logdl):
    """
    :param m: float,
    apparent magnitude
    :param z: float,
    redshift
    :return: M1450, float
    absolute magnitude
    """

    return m +5. - 5. *logdl+ 2.5 * np.log10((1 + z))


def gauss(x, mu, sigma):
    """
    :param x: float
    :param mu: float, mean
    :param sigma: float, standard deviation
    :return: float, gaussian function at point x
    """

    return (np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))) / ( 2.5066282746310002 * sigma)

def flux(m,ZP):
	F0=10**(ZP/2.5)
	return F0*np.exp(-2*np.log(10)*m/5)

def flux_int(m, zp):
    """
    dFmod/dm_mod equation (4.10)
    """
    return -0.9210340371976183 * flux(m, zp)


def flux_int2(m, zp):
    return (np.log(10)) / 2.5 * np.exp(np.log(10) * -1. * (m - zp) / 2.5)


"""

def completude(x: float):

    if x < 25.:
        return 1.0
    else:
        return 0.
"""

def quasar_density(z_mod, magy, Mabs,cv):
    """
    Quasar density function = Prior probability
    :param J: magnitude reference
    :param data: candidate list
    :param np.ndarray z_mod: model redshift
    :param np.ndarray magy: model  apparent magnitude in the Y band
    """

    q_density = np.zeros(((z_mod.shape[0]), (magy.shape[0])))
    for i in range(z_mod.shape[0]):
        q_density[i,:] =(
            cv[i]* luminosity_function(
               Mabs[i], z_mod[i]))

    return q_density


def galaxy_density(z_mod, magy,Mabs,cv):
    """
    Quasar density function = Prior probability
    :param J: magnitude reference
    :param data: candidate list
    :param np.ndarray z_mod: model redshift
    :param np.ndarray magy: model  apparent magnitude in the Y band
    """

    q_density = np.zeros(((z_mod.shape[0]), (magy.shape[0])))
    for i in range(z_mod.shape[0]):
        q_density[i,:] =(
            cv[i]* gal.luminosity_function(
                Mabs, z_mod[i]))

    return q_density

#def quasar_density(z_mod, magy, data, J, const, c, cv, logdl):
#    """
#    Quasar density function = Prior probability
#    :param J: magnitude reference
#    :param data: candidate list
#    :param np.ndarray z_mod: model redshift
#    :param np.ndarray magy: model  apparent magnitude in the Y band
#    """

#    # Calculate the absolute magnitude using the given apparent magnitude and redshift
#    m_abs = m_1450_ob(magy.reshape(-1, 1), z_mod.reshape(1, -1), data, J, const, c, logdl)

#    # Calculate the luminosity function using the absolute magnitude and redshift
#    lum_func = luminosity_function(m_abs, z_mod.reshape(1, -1))

#    # Calculate the quasar density using the pre-computed values and luminosity function
#    q_density = cv.reshape(-1, 1) * lum_func

#    return q_density


def completude_likelihood(obj, magH, z_mod, color, sed, zp, data,num_filt,ref_filter):
    """
    Compute the completeness+ likelihood for each quasar candidate for each template
    :param int obj: quasar candidate index
    :param int nb_obj: number of quasar candidates
    :param np.ndarray magH: model apparent magnitude
    :param np.ndarray z_mod: model redshift
    :param np.ndarray color: input colors calculated from templates
    :param int sed: SED index
    :param  float zp: Zero point
    :param str nom: id of quasar candidate
    :param np.ndarray data: quasar candidates fluxes in z, J and Y bands
    :return: Completeness and likelihood for the marginal probability distribution for the quasars
    """

    gauss_i = np.zeros((num_filt, len(z_mod), len(magH)))

    for f in range(num_filt):
        if f == ref_filter:
            for i in range(len(z_mod)):
                magi = magH
                fluxi = flux(magi, zp)
                gauss_i[f, i, :] = abs(flux_int(magi, zp)) * gauss(data[2 * f + 1 + 2][obj], fluxi,
                                                                        data[2 * f + 2 + 2][obj])
        elif(f<ref_filter):
            nn=f+2
            h_mi = color[len(z_mod) * sed:len(z_mod) * (sed + 1), nn]
            for i in range(len(z_mod)):
                    magi = np.around(-1.0 * h_mi[i] + magH[:], 3)
                    fluxi = flux(magi, zp)
                    #print(f"f={f}, nn={nn}, i={i}, result={result}")
                    gauss_i[f,i, :] = abs(flux_int(magi, zp)) * gauss(data[2 * f + 1+ 2][obj], fluxi, data[2 * f + 2+ 2 ][obj])
        elif((f>ref_filter)):
            nn=f+2-1
            h_mi = color[len(z_mod) * sed:len(z_mod) * (sed + 1), nn]
            for i in range(len(z_mod)):
                    magi = np.around(-1.0 * h_mi[i] + magH[:], 3)
                    fluxi = flux(magi, zp)
                    gauss_i[f,i, :] = abs(flux_int(magi, zp)) * gauss(data[2 * f + 1 + 2][obj], fluxi, data[2 * f + 2 + 2][obj])
    I1 = np.prod(gauss_i, axis=0)
    return I1
