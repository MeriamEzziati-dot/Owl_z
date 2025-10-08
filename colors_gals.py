import os
import warnings
import time
import yaml

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from colors.HZ_colors import *
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
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

    f_lam=data[1]/(1+z)
    return data[0] * (1 + z), f_lam
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

a_yaml_file = open("config.yaml")


if __name__ == "__main__":
    parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    c = float(parsed_yaml_file["light_speed"])
    const = float(parsed_yaml_file["const"])
    start = time.time()
    # Path management
    limag=parsed_yaml_file['LIM_mag']
    path_parent = str(parsed_yaml_file['workdir'])
    template_path_lt = str(parsed_yaml_file['templates_GAL_dir'])
    filter_path = str(parsed_yaml_file['filters_dir'])
    colors_path = str(parsed_yaml_file['colors_result_path'])
    filters_list = str(parsed_yaml_file['filters_list'])
    templates_lt_list = str(parsed_yaml_file['templates_GAL'])
    filters = np.genfromtxt(path_parent + filters_list, dtype=str)
    arr = np.genfromtxt(path_parent + templates_lt_list, dtype=str)
    ref_filter = parsed_yaml_file["reference_filt"]
    num_filt = len(filters)
    ref=np.genfromtxt(path_parent + filter_path+filters [int(ref_filter)]).T

    # arr=np.delete(arr,list(arr).index('L4_7.txt'))

    z = parsed_yaml_file["redshift_g"]
    Z = np.arange(z[0], z[1], z[2])
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # flat cosmology
    my = parsed_yaml_file["magY"]
    Y_model = np.arange(my[0], my[1], my[2])
    file = np.zeros((int(num_filt)+2, len(Z) * len(arr)))
    # Your existing loop to populate 'file'
    for i in range(len(arr)):
        for j in range(len(Z)):
            file[0][j + i * len(Z)] = round(int(i + 1), 0)
            file[1][j + i * len(Z)] = round(Z[j], 2)

                # Your other calculations here

                # Save the 'file' array into a text file after populating it
    for i in range(len(arr)):
        datum = np.loadtxt(path_parent + template_path_lt + arr[i]).T
        k = 0
        mref = np.zeros(len(Z))

        for f in range(num_filt):

            mobs = np.zeros(len(Z))
            for z in range(len(Z)):
                filter=np.genfromtxt(path_parent + filter_path+filters[f]).T
                scaled = scaling(Z[z], datum)
                mref[z] = colors(scaled, const, ref, c)
                mobs[z]=colors(scaled, const,filter, c)
            print(mref)
            file[2 + k][i * len(Z):(1 + i) * len(Z)]= mobs[:]
            file[2 + k][i * len(Z):(1 + i) * len(Z)][mobs==99] = 999
            k+=1

    np.savetxt(path_parent + colors_path + 'GLX_mags.txt', file.T, fmt='%0.2f', delimiter='\t')


    file = np.zeros((int(num_filt)+1, len(Z) * len(arr)))
    # Your existing loop to populate 'file'
    for i in range(len(arr)):
        for j in range(len(Z)):
            file[0][j + i * len(Z)] = round(int(i + 1), 0)
            file[1][j + i * len(Z)] = round(Z[j], 2)

    for i in range(len(arr)):
        datum = np.loadtxt(path_parent + template_path_lt + arr[i]).T
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

                    file[2 + k][i * len(Z):(1 + i) * len(Z)]=mref[:] - mobs[:]
                    file[2 + k][i * len(Z):(1 + i) * len(Z)][mobs == 99] = 999
                k+=1

    np.savetxt(path_parent + colors_path + 'GLX_colors.txt', file.T, fmt='%0.2f', delimiter='\t')
    end = time.time()
    print('Finished executing QSO colors script in', end - start, ' seconds')

