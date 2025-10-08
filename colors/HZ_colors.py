import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import trapz
from astropy import units as u
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # flat cosmology

def magnitude(m1450, zr):
    """
    :param zr: float
    redshift
    :param m1450: float
    absolute magnitude at 1450
    :return:m
     the observed magnitude at 1450(1+z) Angstroms
    """
    return m1450 - 5 + 5 * np.log10(u.Quantity(cosmo.luminosity_distance(zr), u.parsec).value) - 2.5 * np.log10(
        (zr + 1))

def filter_resamp(filter_yjh, xx):
    yy = interp1d(filter_yjh[0], np.abs(filter_yjh[1]), bounds_error=False, fill_value=0)(xx)
    return yy
def flux_nu(m: float):
    """
    :param m: magnitude in mag
    :return: flux nu density erg/s/cm^2/Hz
    """
    return 3631.0e-23 * 10.0 ** (-0.4 * m)


def f_nu(flux_lambda, l,c):
    """
    :param flux_lambda:  erg/s/cm^2/Angstrom
    :param l: lambda Angstrom
    :return: flux nu  unit erg/s/cm^2/Hz
    """
    return flux_lambda * (l ** 2) / c


def flux_lambda(fl_nu, lam,c):
    """
    :param fl_nu:  float
    Flux in frequency unit  erg/s/cm^2/Hz
    :param lam: float
    wavelength unit Angstrom
    :return: float
    Flux lambda in lambda unit erg/s/cm^2/Angstrom
    """
    return fl_nu * c / lam ** 2


def f_template(file):
    """
    :param file: template
    :return: float,
    flux lambda in the  +10 1450 -10 A  in unit:erg/s/cm^2/Angstrom
    """
    l1450 = np.zeros(2, dtype=int)
    f = file[0, :]
    for i in range(f.shape[0]):
        if f[i] >= 1440.0:
            l1450[0] = i
            break
    for i in range(l1450[0] + 1, f.shape[0]):
        if f[i] >= 1460.0:
            l1450[1] = i
            break
    return np.mean(file[1][l1450[0]:l1450[1]])


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



def m_R(filter_fit, lambdas, F_c, const: float,limag):
    """
    Observed magnitude in a filter
    :param filter_fit: filter transmission
    :param lambdas: wavelenghts
    :param F_c: Flux nu
    :param const: hypothetical constant
    :return: observed magnitude in the filter
    """
    if (simps(F_c * filter_fit * lambdas, 1 / lambdas)  <= 0) or (-2.5 * np.log10(
            (simps(F_c * filter_fit * lambdas, 1 / lambdas) / simps(filter_fit * const * lambdas, 1 / lambdas)))>=limag+1):
        m = 99
    else:
        m = -2.5 * np.log10(
            (simps(F_c * filter_fit * lambdas, 1 / lambdas) / simps(filter_fit * const * lambdas, 1 / lambdas)))
    return m

def compare_spacing(arr1, arr2):
    diff1 = np.diff(arr1)
    diff2 = np.diff(arr2)
    avg_diff1 = np.mean(diff1)
    avg_diff2 = np.mean(diff2)

    if avg_diff1 > avg_diff2:
        return arr2
    else:
        return arr1


def colors(data, const, filt, c,limag):
    """
    :param redshift : redshift
    :param data : template (Lambda,Flux_lambda) !!!!!!!!!!!!!!!!!! already redshifted and scaled!!!!!!!!!!!!
    :param const: arbitrary constant
    :param Y: first filter
    :param J: second filter
    :param H: third filter
    :return: observed magnitudes in the three input filters
    """
    lambda_min=filt[0].min()
    lambda_max=filt[0].max()
    xx=np.linspace(lambda_min,lambda_max,int((lambda_max-lambda_min)/5))
    flux = data[1]
    wave = data[0]

    yy = interp1d(wave, flux, bounds_error=False, fill_value=0)(xx)

    m=(m_R(filter_resamp(filt,xx), xx, f_nu(yy,xx, c), const,limag))

    return m
