
import warnings
import os
import time
import yaml
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simps,trapz
import numpy as np
import yaml
#import model.qso as qso
import model.qso_c as qso
#import model.mlt as mlt
import model.mlt_c as mlt
import subprocess
import argparse
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
parser = argparse.ArgumentParser()
parser.add_argument('integer', type=int, nargs='+')
args = parser.parse_args()
candi_index = args.integer[0]
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
#print(num_filt)
ref = np.genfromtxt(path_parent + filter_path + filters[int(ref_filter)]).T
z = parsed_yaml_file["redshift"]
redshift = np.arange(z[0], z[1], z[2])
my = parsed_yaml_file["magY"]
Y_model = np.arange(my[0], my[1], my[2])
ZP = float(parsed_yaml_file["ZP"])
filters=np.genfromtxt(path_parent + filters_list,dtype=str)
c_light = float(parsed_yaml_file["light_speed"])
const = float(parsed_yaml_file["const"])
cand_list=np.genfromtxt( path_parent+candidates+'log.txt', dtype=str)
MABS_QSO_list=parsed_yaml_file["MABS_QSO_list"]
MABS_GAL_list=parsed_yaml_file["MABS_GAL_list"]

#print('ref_filter='+str(path_parent + filter_path + filters[int(ref_filter)]))
#print('cand= '+str(path_parent+candidates + cand_list[candi_index]))
data_candidates = np.loadtxt( path_parent+candidates + cand_list[candi_index]).T
arr = np.genfromtxt(path_parent + MABS_QSO_list, dtype=str)
arrg = np.genfromtxt(path_parent + MABS_GAL_list, dtype=str)

zg = parsed_yaml_file["redshift_g"]
redshiftg = np.arange(zg[0], zg[1], zg[2])
#print("SHAAAAAAAAAAAPE", np.shape(data_candidates))
if __name__ == "__main__":
    #print('*** Calcul des couleurs des LT**')
    #os.system("python colors_stars_MLT.py Arg")
    #print('*** Calcul des couleurs des QSO entre redshift z= ', redshift[0], ' et redshift z= ', round(redshift[-1], 1),
    #      ' ***')
    #os.system("python colors_quasars.py Arg")
    #print('*** Calcul des couleurs des galaxies entre redshift z= ', redshiftg[0], ' et redshift z= ', round(redshiftg[-1], 1),
    #      ' ***')
    #os.system("python colors_gals.py Arg")
    start = time.time()
    M1450_z = []

    # candidates list
    #print('*** Bayesian selection START with 21 SEDs ***')
    nbsed = 21  # Number of SEDs

    nb_obj_q = len(data_candidates[0])  # number of quasar candidates
    ids = data_candidates[0]
    l = data_candidates[1]
    b = data_candidates[2]
    #print(l,b)

    # templates
    ref_filter = parsed_yaml_file["reference_filt"]

    colors_template = np.loadtxt(path_parent + colors_path + 'QSO_colors.txt')
    J = np.genfromtxt(path_parent + filter_path + filters[ref_filter]).T
    Wqf = np.zeros((nbsed, nb_obj_q), dtype=float)  # Marginal probability distribution for quasars

    #print("** Calcul des poids pondérés des QSO **")
    SW=parsed_yaml_file['SW']
    cv = cosmo.differential_comoving_volume(redshift).value * ((np.pi / 180.) ** 2*SW/3)
    logdl = np.log10(u.Quantity(cosmo.luminosity_distance(redshift), u.parsec).value)

    ###############################################################""
    # QSO
    ################################################################
    start0 = time.time()
    print(f">>>1 {start0-start}")
    for sed in range(nbsed):
        #print('SED ' + str(sed + 1) + ' / ' + str(nbsed))
        
        Mabs = np.loadtxt(path_parent + colors_path + arr[sed]).T
        
        P=qso.quasar_density(redshift, Y_model, Mabs, cv)

        for obj in range(nb_obj_q):
            #P=qso.quasar_density(redshift, Y_model, Mabs, cv)
            W=qso.completude_likelihood(obj, Y_model, redshift, colors_template,sed, ZP, data_candidates, num_filt, ref_filter)
            Wqf[sed][obj] =simps(simps(P*W,Y_model),redshift)


    start2 = time.time()
    print(f">>>completude_likehood {start2-start0}")
    MLT_colors = np.genfromtxt(path_parent + str(parsed_yaml_file['LT_dens']), delimiter=None, dtype=None,
                               encoding='ascii')
    spts_M = np.genfromtxt(path_parent + str(parsed_yaml_file['SptM']), dtype=str)
    spts_L = np.genfromtxt(path_parent + str(parsed_yaml_file['SptL']), dtype=str)
    spts_T = np.genfromtxt(path_parent + str(parsed_yaml_file['SptT']), dtype=str)

    MY = MLT_colors["f2"]
    spt = MLT_colors["f0"]
    n = MLT_colors["f1"]
    MYabs = MY[-1]
    nsol = n

    warnings.filterwarnings("ignore")

    SWrad = mlt.surface(parsed_yaml_file['SW'])

    Yfirst = 0.5 * (Y_model[:-1] + Y_model[1:])
    nb_obj = nb_obj_q
    ########################################################################"
    #M
    ########################################################################"

    colors_M = np.loadtxt(path_parent + colors_path + 'M_colors.txt').T
    zmY_BD_M = colors_M[0]
    nb_BD_M = len(zmY_BD_M)

    colors_L = np.loadtxt(path_parent + colors_path + 'L_colors.txt').T

    zmY_BD_L = colors_L[0]
    nb_BD_L = len(zmY_BD_L)

    colors_T = np.loadtxt(path_parent + colors_path + 'T_colors.txt').T
    zmY_BD_T = colors_T[0]
    nb_BD_T = len(zmY_BD_T)
    Wsf_T = np.zeros((nb_obj, nb_BD_T))
    Wsf_M = np.zeros((nb_obj, nb_BD_M))
    Wsf_L = np.zeros((nb_obj, nb_BD_L))
    start3 = time.time()
    print(f">>>2 {start3-start2}")
    
    #print(nb_BD)
    for obj in range(nb_obj):
        lrad = np.deg2rad(l[obj])
        brad = np.deg2rad(b[obj])
        dstar = mlt.d_star(brad, parsed_yaml_file['Z_sol'], parsed_yaml_file['hZ'])
        if (b[obj] < 0):
            nW4, nW4ext = mlt.NWS(Y_model, MY, parsed_yaml_file['EBmV'], parsed_yaml_file['Z_sol'],
                                  parsed_yaml_file['hZ'], lrad,
                                  brad, spt, SWrad, parsed_yaml_file['SW'], parsed_yaml_file['hR'], nsol)
        else:
            nW4, nW4ext = mlt.NWN(Y_model, MY, parsed_yaml_file['EBmV'], parsed_yaml_file['Z_sol'],
                                  parsed_yaml_file['hZ'], lrad,
                                  brad, spt, SWrad, parsed_yaml_file['SW'], parsed_yaml_file['hR'], nsol)
        # nW4, nW4ext = mlt.NW(Y_model, MY, parsed_yaml_file['EBmV'], parsed_yaml_file['Z_sol'], parsed_yaml_file['hZ'], lrad,
        # brad, spt, SWrad, parsed_yaml_file['SW'], parsed_yaml_file['hR'], nsol)
        dnW4ext = mlt.dnWext(Y_model, nW4, nW4ext, spt)
        # print('Object ', obj + 1, ' /', nb_obj)
        for n in range(nb_BD_M):
            WS = mlt.wsJCL(obj, n, Y_model, data_candidates, colors_M, ZP, Yfirst, dnW4ext, spts_M, num_filt,
                           ref_filter)
            Wsf_M[obj, n] = simps(
                WS[0][:] *
                WS[1][:],
                # mlt.ws(obj, n, Y_model, data_candidates, colors_M, ZP, Yfirst, dnW4ext, spts_M,num_filt,ref_filter)[2][obj, n, :],
                Y_model)
        for n in range(nb_BD_L):
            WS = mlt.wsJCL(obj, n, Y_model, data_candidates, colors_L, ZP, Yfirst, dnW4ext, spts_L, num_filt,
                           ref_filter)
            Wsf_L[obj, n] = simps(
                WS[0][:] *
                WS[1][:],
                # mlt.ws(obj, n, Y_model, data_candidates, colors_L, ZP, Yfirst, dnW4ext, spts_L,num_filt,ref_filter)[2][obj, n, :],
                Y_model)
        for n in range(nb_BD_T):
            WS = mlt.wsJCL(obj, n, Y_model, data_candidates, colors_T, ZP, Yfirst, dnW4ext, spts_T, num_filt,
                           ref_filter)
            Wsf_T[obj, n] = simps(
                WS[0][:] *
                WS[1][:] ,Y_model)
            
    start4 = time.time()
    print(f">>>wsJCL {start4-start3}")
                #mlt.ws(obj,
    densM =Wsf_M.mean(1)
    densL =Wsf_L.mean(1)
    densT =Wsf_T.mean(1)




#########################################"" GALAXIES################################""""
    import model.cont_gal as gal

    gseds = 38

    template_path_GAL = str(parsed_yaml_file['templates_GAL_dir'])
    templates_g_list = str(parsed_yaml_file['templates_GAL'])
    Wgf = np.zeros((gseds, nb_obj_q) )  # Marginal probability distribution for quasars

    zg = parsed_yaml_file["redshift_g"]
    redshiftg = np.arange(zg[0], zg[1], zg[2])
    arr_g = np.genfromtxt(path_parent + templates_g_list, dtype=str)
    cvg = cosmo.differential_comoving_volume(redshiftg).value *(np.pi/180)**2* SW/3
    colors_template_g = np.loadtxt(path_parent + colors_path + 'GLX_colors.txt')


#### GALAXIES
    start5 = time.time()
    print(f">>>5 {start5-start4}")
    for sed in range(gseds):
        #print('SED ' + str(sed + 1) + ' / ' + str(gseds))
        M_G = np.loadtxt(path_parent + colors_path + arrg[sed]).T
        s=gal.galaxy_density(Y_model,redshiftg,M_G,cvg)
        for obj in range(nb_obj_q):
            
	    #s=gal.galaxy_density(Y_model,redshiftg,M_G,cvg)
            WG=qso.completude_likelihood(obj, Y_model, redshiftg, colors_template_g,
                                                  sed, ZP, data_candidates,num_filt,ref_filter)
            Wgf[sed][obj]=simps(simps(s *WG, Y_model),redshiftg)
    ws=densT+densM+densL
    start6 = time.time()
    print(f">>>completude_likehood {start6-start5}")
    # M, L, T probabilities separa
     
    pm =densM / (Wgf.mean(0) + ws + Wqf.mean(0))
    pl = densL / (Wgf.mean(0) + ws + Wqf.mean(0))
    pt = densT / (Wgf.mean(0) + ws + Wqf.mean(0))

    pg =Wgf.mean(0)/(Wgf.mean(0)+ws+Wqf.mean(0))
    pq =Wqf.mean(0)/(Wgf.mean(0)+ws+Wqf.mean(0))
    ps =ws/(Wgf.mean(0)+ws+Wqf.mean(0))
    file2 = path_parent+result_path+"TEST_"+str(candi_index)+"_all_glx.txt"

    np.savetxt(file2,np.array([ids,ps,pg,pq,pm,pl,pt]).T)

    print(pq)
    end = time.time()
    print("IT tOOK it",end-start)
