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
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def completude(m):
    """
    Completeness for the four fields fit done by Sarah using Sexstractor
    :param x: magnitude Y
    :param chp: field W1,W2,W3 or W4
    :return: Completeness
    """
    return 1.0

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
def schechter_lf_M(m):
    """
    Calculate the Schechter luminosity function for galaxies.

    Parameters:
        luminosity (float or numpy.ndarray): Luminosity value(s) at which to evaluate the function.
        L_star (float): Characteristic luminosity.
        alpha (float): Faint-end slope.
        phi_star (float): Overall number density.

    Returns:
        float or numpy.ndarray: The value(s) of the Schechter luminosity function at the given luminosity(s).
    """
    # Ensure luminosity is a numpy array for element-wise calculations
    #phi_star= 0.00116649
    #phi_star=0.001e-6
    phi_star= 0.001
    alpha=-1.22378533
    #alpha=1
    m_star=-20.7619195

    return  ((0.4*np.log(10)) *(phi_star) * (10**(0.4*(m_star-m)))**(alpha+1) * np.exp(-10**(0.4*(m_star-m))))
def luminosity_function(magy,z_mod,data, J, const,c):
    """ Computes the luminosity function for a quasar z>6) using a LF for a quasar at z=6
    prends magY et z
    """
    logdl= np.log10(u.Quantity(cosmo.luminosity_distance(z_mod), u.parsec).value)

    return schechter_lf_M( m_1450_ob(magy[:], z_mod[i], data, J, const,c,logdl[i]))



def galaxy_density(magy, z_mod, Mabs,cv):
    """
    Quasar density function = Prior probability
    :param J: magnitude reference
    :param data: candidate list
    :param np.ndarray z_mod: model redshift
    :param np.ndarray magy: model  apparent magnitude in the Y band
    """
    q_density = np.zeros(((z_mod.shape[0]), (magy.shape[0])))
    for i in range(z_mod.shape[0]):
        q_density[i,:] =cv[i]* schechter_lf_M(Mabs[i])
    return q_density

