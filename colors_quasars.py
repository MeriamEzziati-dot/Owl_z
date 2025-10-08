import os
import warnings
import time
import yaml
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
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
def scaling( z, data):
    new_template = data[1]/(1+z)
    return data[0] * (1 + z), new_template
warnings.filterwarnings("ignore")
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # flat cosmology
def colors(data, const, filt, c):
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

    m=(m_Rr(filter_resamp(filt,xx), xx, f_nu(yy,xx, c), const))

    return m

# path= os.path.dirname(os.getcwd())
a_yaml_file = open("config.yaml")

if __name__ == "__main__":

    parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    c = float(parsed_yaml_file["light_speed"])
    const = float(parsed_yaml_file["const"])
    start = time.time()
    # Path management
    path_parent = str(parsed_yaml_file['workdir'])
    template_path_qso = str(parsed_yaml_file['templates_QSO_dir'])
    filter_path = str(parsed_yaml_file['filters_dir'])

    filters_list = str(parsed_yaml_file['filters_list'])
    templates_qso_list = str(parsed_yaml_file['templates_QSO'])
    filters = np.genfromtxt(path_parent + filters_list, dtype=str)
    ref_filter = parsed_yaml_file["reference_filt"]
    num_filt = len(filters)
    ref=np.genfromtxt(path_parent + filter_path+filters [int(ref_filter)]).T
    my = parsed_yaml_file["magY"]
    Y_model = np.arange(my[0], my[1], my[2])
    print(Y_model.shape)
    print(filters [int(ref_filter)])
    arr = np.genfromtxt(path_parent + templates_qso_list, dtype=str)
    colors_path = str(parsed_yaml_file['colors_result_path'])
    # general parameters

    z = parsed_yaml_file["redshift"]
    Z = np.arange(z[0], z[1], z[2])
    print(arr)
    # Euclid filters import

    file = np.zeros((int(num_filt)+2, len(Z) * len(arr)))
    # Your existing loop to populate 'file'
    for i in range(len(arr)):
        for j in range(len(Z)):
            file[0][j + i * len(Z)] = round(int(i + 1), 0)
            file[1][j + i * len(Z)] = round(Z[j], 2)

                # Your other calculations here

                # Save the 'file' array into a text file after populating it
    for i in range(len(arr)):
        datum = np.loadtxt(path_parent + template_path_qso + arr[i]).T
        k = 0


        for f in range(num_filt):
            mref = np.zeros(len(Z))
            mobs = np.zeros(len(Z))
            for z in range(len(Z)):

                filter=np.genfromtxt(path_parent + filter_path+filters[f]).T
                scaled = scaling( Z[z], datum)
                mref[z] = colors(scaled, const, ref, c)
                mobs[z]=colors(scaled, const,filter, c)
            file[2 + k][i * len(Z):(1 + i) * len(Z)][mobs==99] = 999
            file[2 + k][i * len(Z):(1 + i) * len(Z)]= mobs[:]
            k+=1

    np.savetxt(path_parent + colors_path + 'QSO_mags.txt', file.T, fmt='%0.2f', delimiter='\t')


    file = np.zeros((int(num_filt)+1, len(Z) * len(arr)))
    # Your existing loop to populate 'file'
    for i in range(len(arr)):
        for j in range(len(Z)):
            file[0][j + i * len(Z)] = round(int(i + 1), 0)
            file[1][j + i * len(Z)] = round(Z[j], 2)
                # Your other calculations here

                # Save the 'file' array into a text file after populating it

    for i in range(len(arr)):
        datum = np.loadtxt(path_parent + template_path_qso + arr[i]).T
        k = 0
        mref = np.zeros(len(Z))
        print(i+1)
        for f in range(num_filt):
            if f == ref_filter:
                continue
            else:
                mobs = np.zeros(len(Z))
                for z in range(len(Z)):

                    filter=np.genfromtxt(path_parent + filter_path+filters[f]).T
                    scaled = scaling( Z[z], datum)
                    mref[z] = colors(scaled, const, ref, c)
                    mobs[z]=colors(scaled, const,filter, c)
                    file[2 + k][i * len(Z):(1 + i) * len(Z)] = mref[:] - mobs[:]
                    file[2 + k][i * len(Z):(1 + i) * len(Z)][mobs == 99] = 999
                k+=1

    np.savetxt(path_parent + colors_path + 'QSO_colors.txt', file.T, fmt='%0.2f', delimiter='\t')
    end = time.time()
    print('Finished executing QSO colors script in', end - start, ' seconds')

