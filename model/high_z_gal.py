import numpy as np

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


warnings.filterwarnings("ignore")

cosmo = FlatLambdaCDM(H0=67.3, Om0=0.315)



def luminosity_function(z,Muv):
    """double power luminosity function by magnitudes."""
    Mstar = -21.03 + 0.49*(z - 6)
    phiStar =10**( -3.52 - 0.00*(z - 6))
    alpha = -1.99 - 0.09*(z - 6)
    beta = -4.92 + 0.09*(z - 6)
    parta = 10 ** (0.4 * (alpha + 1.0) * (Muv - Mstar))
    partb = 10 ** (0.4 * (beta + 1.0) * (Muv - Mstar))
    return phiStar/(parta + partb)
"""
z=np.linspace(5,10,10)
magnitude =np.linspace(-25,-16,100)
import matplotlib.pyplot as plt
plt.title('Galaxies')
for i in range(len(z)):
    plt.plot(magnitude,luminosity_function(z[i],magnitude),label='z= '+str(z[i]))
plt.xlabel('Abs mag')
plt.ylabel('LF')
#plt.legend()
plt.show()
"""

def luminosity_function_q(x, z):
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
"""
plt.title('Quasars')
for i in range(len(z)):
    plt.plot(magnitude,luminosity_function_q(magnitude,z[i]),label='z= '+str(z[i]))
plt.xlabel('Abs mag')
plt.ylabel('LF')
plt.show()
"""

