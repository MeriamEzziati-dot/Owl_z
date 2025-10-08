"""
-----------------------------------------------------
Author : Meriam Ezziati
Date   : 8 April 2022
Cythonized by: Jean-Charles Lambert Mars 2024
------------------------------------------------------

MODULE QSO: Functions to calculate the marginal distribution for the bayesian
selection method for high redshift quasars

"""

cimport cython
from cython.parallel cimport prange,parallel
cimport openmp
import numpy as np
from openmp cimport omp_get_thread_num,omp_get_num_threads

import warnings

#import numpy as np
from scipy import special, interpolate
from astropy.cosmology import FlatLambdaCDM
from decimal import Decimal
from model.inverse_scaling import *
import model.high_z_gal as gal
warnings.filterwarnings("ignore")

cosmo = FlatLambdaCDM(H0=67.3, Om0=0.315)

def gaussian_cdf(x,mu,sigma):
    return 0.5 * (1 + special.erf((x - mu) / (sigma * np.sqrt(2))))

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



def gauss(x, mu, sigma):
    """
    :param x: float
    :param mu: float, mean
    :param sigma: float, standard deviation
    :return: float, gaussian function at point x
    """

    return (np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))) / ( 2.5066282746310002 * sigma)

def flux( m,  ZP):
    F0=10**(ZP/2.5)
    return F0*np.exp(-2*np.log(10)*m/5)




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



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def completude_likelihood(int obj, magH, double[:] z_mod, double[:, :] color, int sed, double zp, double[:, :] data, int num_filt, int ref_filter, double[:] fluxlimite):
    """
    Compute the completeness+ likelihood for each quasar candidate for each template
    :param int obj: quasar candidate index
    :param np.ndarray magH: model apparent magnitude
    :param np.ndarray z_mod: model redshift
    :param np.ndarray color: input colors calculated from templates
    :param int sed: SED index
    :param  float zp: Zero point
    :param np.ndarray data: quasar candidates fluxes in z, J and Y bands
    :param int num_filt: number of filters
    :param int ref_filter: reference filter index
    :param np.ndarray fluxlimite: flux limits
    :return: Completeness and likelihood for the marginal probability distribution for the quasars
    """


    # indexes
    cdef Py_ssize_t f,i,j, l_z_mod,l_magH,nn
    l_z_mod = len(z_mod)
    l_magH = len(magH)
    gauss_i = np.zeros((num_filt, l_z_mod, l_magH))

    for f in range(num_filt):
        F = data[2 * f + 3][obj]

        errf = data[2 * f + 4][obj]

        if f == ref_filter:
            f_int2 = -0.9210340371976183 * flux(magH, zp)
            #LAM = 1.0 / (1.0 - gaussian_cdf(0, flux(magH, zp), errf))
            LAM=1.0
            gauss_i[f, :, :] = LAM * np.abs(f_int2) * gauss(F, flux(magH, zp), errf)
        else:
            if f < ref_filter:
                nn = f + 2
            else:
                nn = f + 1
            h_mi = color[l_z_mod * sed:l_z_mod * (sed + 1), nn]
            for i in range(l_z_mod):
                magi = np.around(-1.0 * h_mi[i] + magH[:], 3)
                if (h_mi[i] == 999.0  ):
                    fluxi = 1e-60
                else:
                    fluxi = flux(magi, zp)
                LAM=1.0
                #LAM=1.0 / (1.0 - gaussian_cdf(0, fluxi, errf))
                gauss_i[f, i, :] = LAM * gauss(F, fluxi, errf)

    cdef double[:, :] I1 = np.prod(gauss_i, axis=0)
    return I1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def completude_likelihood_old(int obj, magH, double[:] z_mod, double[:, :] color, int sed, double zp, double[:, :] data, int num_filt, int ref_filter, double[:] fluxlimite):
    """
    Compute the completeness+ likelihood for each quasar candidate for each template
    :param int obj: quasar candidate index
    :param np.ndarray magH: model apparent magnitude
    :param np.ndarray z_mod: model redshift
    :param np.ndarray color: input colors calculated from templates
    :param int sed: SED index
    :param  float zp: Zero point
    :param np.ndarray data: quasar candidates fluxes in z, J and Y bands
    :param int num_filt: number of filters
    :param int ref_filter: reference filter index
    :param np.ndarray fluxlimite: flux limits
    :return: Completeness and likelihood for the marginal probability distribution for the quasars
    """


    # indexes
    cdef Py_ssize_t f,i,j, l_z_mod,l_magH,nn
    l_z_mod = len(z_mod)
    l_magH = len(magH)
    gauss_i = np.zeros((num_filt, l_z_mod, l_magH))

    for f in range(num_filt):
        F = data[2 * f + 3][obj]
        errf =data[2 * f + 4][obj]
        SNR=F/errf

        if f == ref_filter:
            f_int2 = -0.9210340371976183 * flux(magH, zp)
            LAM = 1.0 / (1.0 - gaussian_cdf(0, flux(magH, zp), errf))
            gauss_i[f, :, :] = LAM * np.abs(f_int2) * gauss(F, flux(magH, zp), errf)
        else:
            if f < ref_filter:
                nn = f + 2
            else:
                nn = f + 1
            h_mi = color[l_z_mod * sed:l_z_mod * (sed + 1), nn]
            for i in range(l_z_mod):
                magi = np.around(-1.0 * h_mi[i] + magH[:], 3)
                if (h_mi[i] == 999.0  ):
                    fluxi = 1e-60
                else:
                    fluxi = flux(magi, zp)

                LAM = 1.0 / (1.0 - gaussian_cdf(0, fluxi, errf))

                if SNR<1:
                    gauss_i[f, i, :] = LAM * gaussian_cdf(fluxlimite[f], fluxi, errf)
                else:
                    gauss_i[f, i, :] = LAM * gauss(F, fluxi, errf)

    cdef double[:, :] I1 = np.prod(gauss_i, axis=0)
    return I1
