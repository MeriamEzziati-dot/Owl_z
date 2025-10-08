
import warnings
import os
import time
import yaml
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simps,trapz
import numpy as np
import yaml
import model.qso as qso
import model.mlt as mlt
import subprocess
import argparse
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import model.inverse_scaling as ins
from scipy.interpolate import interp1d

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
def filter_resamp(filter_yjh, xx):
    yy = interp1d(filter_yjh[0], filter_yjh[1], bounds_error=False, fill_value=0)(xx)
    return yy
a_yaml_file = open("config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
filters_list=str(parsed_yaml_file['filters_list'])
filt_num=len(filters_list)
path_parent = str(parsed_yaml_file['workdir'])
candidates = (str(parsed_yaml_file['candidates']))
colors_path = str(parsed_yaml_file['colors_result_path'])
filter_path = str(parsed_yaml_file['filters_dir'])
result_path = str(parsed_yaml_file['probas_result_path'])
template_path_QSO = str(parsed_yaml_file['templates_QSO_dir'])
templates_qso_list = str(parsed_yaml_file['templates_QSO'])
filters = np.genfromtxt(path_parent + filters_list, dtype=str)
ref_filter = parsed_yaml_file["reference_filt"]
num_filt = len(filters)
print(num_filt)
ref = np.genfromtxt(path_parent + filter_path + filters[int(ref_filter)]).T
z = parsed_yaml_file["redshift"]
redshift = np.arange(z[0], z[1], z[2])
my = parsed_yaml_file["magY"]
Y_model = np.arange(my[0], my[1], my[2])
ZP = float(parsed_yaml_file["ZP"])
filters=np.genfromtxt(path_parent + filters_list,dtype=str)
c_light = float(parsed_yaml_file["light_speed"])
const = float(parsed_yaml_file["const"])

print('ref_filter='+str(path_parent + filter_path + filters[int(ref_filter)]))
arr = np.genfromtxt(path_parent + templates_qso_list, dtype=str)
nbsed=len(arr)
zg = parsed_yaml_file["redshift_g"]
redshiftg = np.arange(zg[0], zg[1], zg[2])
J = np.genfromtxt(path_parent + filter_path + filters[ref_filter]).T
xx = np.arange(1000, 25000, 5)

Hf= filter_resamp(J, xx)
if __name__ == "__main__":
    logdl = np.log10(u.Quantity(cosmo.luminosity_distance(redshift), u.parsec).value)

    for sed in range(len(arr)):
        template = np.loadtxt(path_parent + template_path_QSO + arr[sed]).T
        M=np.zeros((len(redshift),len(Y_model)))
        for zz in range(len(redshift)):
            for m in range(len(Y_model)):
                M[zz][m] =ins.m_1450_ob(Y_model[m], redshift[zz], template, Hf, const, c_light,logdl[zz],xx)
                #print(M[zz][m])
        np.savetxt(path_parent + colors_path +str(int(sed))+'_M_abscorrected.txt',M.T)


    template_path_GAL = str(parsed_yaml_file['templates_GAL_dir'])
    templates_g_list = str(parsed_yaml_file['templates_GAL'])

    zg = parsed_yaml_file["redshift_g"]
    redshiftg = np.arange(zg[0], zg[1], zg[2])
    arr_g = np.genfromtxt(path_parent + templates_g_list, dtype=str)
    logdl=np.log10(u.Quantity(cosmo.luminosity_distance(redshiftg), u.parsec).value)
    for sed in range(58):
        template = np.loadtxt(path_parent + template_path_GAL + arr_g[sed]).T
        M=np.zeros((len(redshiftg),len(Y_model)))
        for zz in range(len(redshiftg)):
            for m in range(len(Y_model)):
                M[zz][m] =ins.m_1450_ob_G(Y_model[m], redshiftg[zz], template, Hf, const, c_light,logdl[zz],xx)
        np.savetxt(path_parent + colors_path +str(int(sed))+'_M_abs_gals.txt',M.T)
