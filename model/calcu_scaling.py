"""
This code computes the M1540 from the apparent magnitude
------------------Inverse Scaling-----------------------

"""


import warnings
import os
import time
import yaml
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simps,trapz
import numpy as np
import yaml

import subprocess
import argparse
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
mH=21

z=7.2
SEDS=8
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
    if ((simps(F_c * filter_fit * lambdas, 1 / lambdas) / simps(filter_fit * const * lambdas, 1 / lambdas)) <= 0) or (-2.5 * np.log10(
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


def colors(redshift, data, const, filt, c,limag):
    """
    :param redshift : redshift
    :param data : template (Lambda,Flux_lambda)
    :param const: arbitrary constant
    :param Y: first filter
    :param J: second filter
    :param H: third filter
    :return: observed magnitudes in the three input filters
    """
    lambda_min=filt[0].min()
    lambda_max=filt[0].max()
    xx=np.linspace(lambda_min,lambda_max,int((lambda_max-lambda_min)/5))
    m_abs = -25
    flux = data[1]
    wave = data[0]
    ftemp = data[1][(data[0] > 1440.) & (data[0] < 1460.)].mean()  # goes to cluster

    # for i in range(len(redshift)):

    mobs = magnitude(m_abs, redshift)
    Flux_nu = flux_nu(mobs)
    Flux_lambda = flux_lambda(Flux_nu, 1450. * (1 + redshift), c)
    S = Flux_lambda / ftemp
    new_template = flux * S
    yy = interp1d(wave * (1 + redshift), new_template, bounds_error=False, fill_value=0)(xx)
    m=(m_R(filter_resamp(filt,xx), xx, f_nu(yy,xx, c), const,limag))

    return m

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

a_yaml_file = open("config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
filters_list=str(parsed_yaml_file['filters_list'])
filt_num=len(filters_list)
path_parent = str(parsed_yaml_file['workdir'])
candidates = (str(parsed_yaml_file['candidates']))
colors_path = str(parsed_yaml_file['colors_result_path'])
filter_path = str(parsed_yaml_file['filters_dir'])
result_path = str(parsed_yaml_file['probas_result_path'])
filters = np.genfromtxt(path_parent + filters_list, dtype=str)
ref_filter = parsed_yaml_file["reference_filt"]
num_filt = len(filters)
ref = np.genfromtxt(path_parent + filter_path + filters[int(ref_filter)]).T

ZP = float(parsed_yaml_file["ZP"])
filters=np.genfromtxt(path_parent + filters_list,dtype=str)
c_light = float(parsed_yaml_file["light_speed"])
const = float(parsed_yaml_file["const"])
template_path_qso = str(parsed_yaml_file['templates_QSO_dir'])
templates_qso_list = str(parsed_yaml_file['templates_QSO'])
arr = np.genfromtxt(path_parent + templates_qso_list, dtype=str)


colors_template = np.loadtxt(path_parent + colors_path + 'QSO_colors.txt')
from astropy import units as u
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import trapz




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
J = np.genfromtxt(path_parent + filter_path + filters[ref_filter]).T

####ATTENTIOOON
def m_1450_ob(my, z, data, J, const, c):
    F_c = fit(data[0] * (1 + z), data[1], J[0])  # f_lambda resampled
    F_nu = f_nu(F_c, J[0], c)  # f_nu resampled
    my=20.75
    mm = m_R(J[1], J[0], F_nu, const)
    S = 10 ** (((my - mm) / -2.5))
    print(S)
    f_lam= S* data[1]
    return data[0]*(1+z),f_lam
templates_qso_list = str(parsed_yaml_file['templates_QSO'])

data = np.loadtxt(path_parent + template_path_qso + arr[SEDS]).T
spectrum = np.loadtxt('/home/mezziati/Documents/These/LAM/spectraz>7/J1342+0928combined_Banados_2018_Natur_553_473B.dat').T
fluxes_n = [9.3756e-29, 1.8197e-28, 3.5645e-28,3.5318e-28,3.1046e-28,3.2810e-28 ]
errorors_n = [ 1.6407e-29, 1.8436e-29, 6.5660e-30,3.9035e-29,4.2891e-29,8.7634e-29]
lambdas = [ 10810.00, 13280.00,17840,23800,33526.00,46028.00]
fluxes = np.array(fluxes_n) * 3e18 / np.array(lambdas) ** 2
errors = np.array(errorors_n) * 3e18 / np.array(lambdas) ** 2
plt.scatter(lambdas, fluxes, color='black', zorder=10)
magnitude_23_3 = 23.3
flux_density_23_3 = 3631.0e-23 * 10.0 ** (-0.4 * magnitude_23_3)
upper_limit=flux_density_23_3*3e18 / np.array(9290.00) ** 2
print(flux_density_23_3)
#plt.plot(spectrum[0], (spectrum[3]), label='Spectrum') # Plot spectrum first
plt.plot(m_1450_ob(mH,z, data, J, const, c_light)[0], m_1450_ob(mH, z, data, J, const, c_light)[1], label='best fit SED')
plt.title('Banados QSO z=7.54, z_out=7.6')
# Plot errors on flux points
plt.errorbar(lambdas, fluxes, yerr=errors, fmt='s', color='black', zorder=11, label='original photometry')
plt.xlim(8500,47050.00)
plt.xlabel('Wavelength')
plt.ylabel('Flux')
#plt.ylim(-1e-18,1.5*1e-17)
plt.annotate('', xy=(9000, upper_limit+0.2e-17), xytext=(9000, upper_limit),
             arrowprops=dict(arrowstyle='->', color='black',linewidth=2, mutation_scale=20), zorder=9)
plt.legend()
plt.show()