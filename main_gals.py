
import warnings
import os
import time
import yaml
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simps, trapz
import numpy as np
import yaml
import model.qso_c as qso
import model.mlt_c as mlt
import subprocess
import argparse
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
parser = argparse.ArgumentParser()
parser.add_argument('integer', type=int, nargs='+')
args = parser.parse_args()
candi_index = args.integer[0]
a_yaml_file = open("config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
filters_list = str(parsed_yaml_file['filters_list'])
filt_num = len(filters_list)
path_parent = str(parsed_yaml_file['workdir'])
candidates = (str(parsed_yaml_file['candidates']))
colors_path = str(parsed_yaml_file['colors_result_path'])
filter_path = str(parsed_yaml_file['filters_dir'])
result_path = str(parsed_yaml_file['probas_result_path'])
filters = np.genfromtxt(path_parent + filters_list, dtype=str)
ref_filter = parsed_yaml_file["reference_filt"]
num_filt = len(filters)
ref = np.genfromtxt(path_parent + filter_path + filters[int(ref_filter)]).T
z = parsed_yaml_file["redshift"]
redshift = np.arange(z[0], z[1], z[2])
my = parsed_yaml_file["magY"]
Y_model = np.arange(my[0], my[1], my[2])
ZP = float(parsed_yaml_file["ZP"])
filters = np.genfromtxt(path_parent + filters_list, dtype=str)
c_light = float(parsed_yaml_file["light_speed"])
const = float(parsed_yaml_file["const"])
cand_list = np.genfromtxt(path_parent + candidates + 'log.txt', dtype=str)
MABS_QSO_list = parsed_yaml_file["MABS_QSO_list"]
MABS_GAL_list = parsed_yaml_file["MABS_GAL_list"]
data_candidates = np.loadtxt(path_parent + candidates + cand_list[candi_index]).T
print('File sued:', cand_list[candi_index])
arr = np.genfromtxt(path_parent + MABS_QSO_list, dtype=str)
arrg = np.genfromtxt(path_parent + MABS_GAL_list, dtype=str)
print(filters)
print('HERE number of filters', len(filters))
zg = parsed_yaml_file["redshift_g"]
redshiftg = np.arange(zg[0], zg[1], zg[2])
SW = parsed_yaml_file['SW']

ref_filter = parsed_yaml_file["reference_filt"]

colors_template = np.loadtxt(path_parent + colors_path + 'QSO_colors.txt')
limag = parsed_yaml_file["LIM_mag"]
print(limag)
fluxlimite = 10 ** (ZP / 2.5) * np.exp(-2 * np.log(10) * np.array(limag) / 5)
ages=np.loadtxt(path_parent+'/data/model_data/GLX/BC03/ages_g.txt').T[1]
#print(ages)
print(fluxlimite)
if __name__ == "__main__":

    start = time.time()
   
    
    T0= [0, 6+1]
    T1= [7, 12+1]
    T2= [13, 19+1]
    T3= [20, 23+1]
    T4= [24, 30+1]
    T5= [31, 41+1]
    T6= [42, 51+1]
    T7= [52, 56+1]
    T8= [57, 58]
    
    L0= [0, 5+1]
    L1= [6, 24+1]
    L2= [25, 32+1]
    L3= [33, 36+1]
    L4= [37, 46+1]
    L5= [47, 59+1]
    L6= [60, 69+1]
    L7= [70, 73+1]
    L8= [74, 77]
    L9= [77, 84]
    """

    L0 = [0, 7]
    L1 = [L0[1], 36 + L0[1]]
    L2 = [L1[1], 23 + L1[1]]
    L3 = [L2[1], 9 + L2[1]]
    L4 = [L3[1], 14 + L3[1]]
    L5 = [L4[1], 26 + L4[1]]
    L6 = [L5[1], 27 + L5[1]]
    L7 = [L6[1], 14 + L6[1]]
    L8 = [L7[1], 13 + L7[1]]
    L9 = [L8[1], 20 + L8[1]]
    """
    all_listsL = [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9]
    """
    T0 = [0, 18]
    T1 = [T0[1], 17 + T0[1]]
    T2 = [T1[1], 18 + T1[1]]
    T3 = [T2[1], 7 + T2[1]]
    T4 = [T3[1], 13 + T3[1]]
    T5 = [T4[1], 22 + T4[1]]
    T6 = [T5[1], 10 + T5[1]]
    T7 = [T6[1], 8 + T6[1]]
    T8 = [T7[1], 4 + T7[1]]
    """
    all_listsT = [T0, T1, T2, T3, T4, T5, T6, T7, T8]

    M1450_z = []
    gseds = 58
    nbsed = 18  # Number of SEDs
    nb_obj = len(data_candidates[0])  # number of quasar candidates
    print('NB OF QSO', nb_obj)
    ids = data_candidates[0]
    ra = data_candidates[1]
    dec = data_candidates[2]

# Convert RA and Dec to Galactic coordinates (l, b)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    l = coords.galactic.l.deg
    b = coords.galactic.b.deg


    J = np.genfromtxt(path_parent + filter_path + filters[ref_filter]).T
    Wqf = np.zeros((nbsed, nb_obj), dtype=float)  # Marginal probability distribution for quasars

    cv = cosmo.differential_comoving_volume(redshift).value * ((np.pi / 180.) ** 2)
    logdl = np.log10(u.Quantity(cosmo.luminosity_distance(redshift), u.parsec).value)
    maxi_densites, indices_maxi = np.zeros((nbsed, len(data_candidates[0]))), np.zeros((nbsed, len(data_candidates[0])))

    densites = np.zeros((nbsed, len(data_candidates[0])))

    start0 = time.time()
    print(f">>>1 {start0 - start}")
    z_index, m_index = [], []

    MABS = np.zeros((nbsed, len(redshift), len(Y_model)))
    for obj in range(nb_obj):
        print('OBJECT= ', obj)
        for sed in range(nbsed):
            MABS[sed] = np.loadtxt(path_parent + colors_path + arr[sed]).T

            P = qso.quasar_density(redshift, Y_model, MABS[sed], cv)
            W = qso.completude_likelihood(obj, Y_model, redshift, colors_template, sed, ZP, data_candidates, num_filt,
                                          ref_filter, fluxlimite)
            Wqf[sed][obj] = simps(simps(P * W, Y_model), redshift)

            indices_maxi[sed, obj] = np.argmax((simps(P * W, Y_model)))
            result = P * W
            maxi_densites[sed, obj] = redshift[int(indices_maxi[sed, obj])]
            densites[sed, obj] = simps(P * W, Y_model)[int(indices_maxi[sed, obj])]
    max_indices = np.argmax(densites, axis=0)
    indexstar = 0
    maxi = (densites.T[indexstar]).max()
    L = np.zeros(nb_obj)
    for i in range(nb_obj):
        L[i] = maxi_densites[int(max_indices[i]), i]
    start2 = time.time()
    print(f">>>Start BD CALCULATION {start2 - start0}")

    warnings.filterwarnings("ignore")

    SWrad = mlt.surface(parsed_yaml_file['SW'])

    M_colors = np.genfromtxt(path_parent + str(parsed_yaml_file['M_dens']), delimiter=None, dtype=None,
                             encoding='ascii')
    L_colors = np.genfromtxt(path_parent + str(parsed_yaml_file['L_dens']), delimiter=None, dtype=None,
                             encoding='ascii')
    T_colors = np.genfromtxt(path_parent + str(parsed_yaml_file['T_dens']), delimiter=None, dtype=None,
                             encoding='ascii')

    MJL = L_colors["f2"]
    JHL = L_colors['f6']
    MYL = MJL -JHL
    sptL = L_colors["f0"]
    nL = L_colors["f1"]
    nsolL = nL
    MJT = T_colors["f2"]
    JHT = T_colors['f6']
    MYT = MJT -JHT
    sptT = T_colors["f0"]
    nT = T_colors["f1"]
    nsolT = nT
    MJM = M_colors["f2"]
    JHM = M_colors['f6']
    MYM = MJM -JHM

    sptM = M_colors["f0"]
    nM = M_colors["f1"]
    nsolM = nM

    spts_L = np.genfromtxt(path_parent + str(parsed_yaml_file['SptL']), dtype=str)
    spts_M = np.genfromtxt(path_parent + str(parsed_yaml_file['SptM']), dtype=str)
    spts_T = np.genfromtxt(path_parent + str(parsed_yaml_file['SptT']), dtype=str)
    Yfirst = 0.5 * (Y_model[:-1] + Y_model[1:])
    ########################################################################"
    # M
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
    print(f">>>2 {start3 - start2}")
    maxi_densites_T, indices_maxi_T = np.zeros((nb_BD_T, nb_obj)), np.zeros((nb_BD_T, nb_obj))

    densites_T = np.zeros((nb_BD_T, nb_obj))
    for obj in range(nb_obj):
        print('OBJECT MLT= ', obj)
        lrad = np.deg2rad(l[obj])
        brad = np.deg2rad(b[obj])
        dstar = mlt.d_star(brad, parsed_yaml_file['Z_sol'], parsed_yaml_file['hZ'])
        if (b[obj] < 0):
            nW4M, nW4extM = mlt.NWS(Y_model, MYM, parsed_yaml_file['EBmV'], parsed_yaml_file['Z_sol'],
                                    parsed_yaml_file['hZ'], lrad,
                                    brad, sptM, SWrad, parsed_yaml_file['SW'], parsed_yaml_file['hR'], nsolM)
            nW4L, nW4extL = mlt.NWS(Y_model, MYL, parsed_yaml_file['EBmV'], parsed_yaml_file['Z_sol'],
                                    parsed_yaml_file['hZ'], lrad,
                                    brad, sptL, SWrad, parsed_yaml_file['SW'], parsed_yaml_file['hR'], nsolL)
            nW4T, nW4extT = mlt.NWS(Y_model, MYT, parsed_yaml_file['EBmV'], parsed_yaml_file['Z_sol'],
                                    parsed_yaml_file['hZ'], lrad,
                                    brad, sptT, SWrad, parsed_yaml_file['SW'], parsed_yaml_file['hR'], nsolT)
        else:

            nW4M, nW4extM = mlt.NWN(Y_model, MYM, parsed_yaml_file['EBmV'], parsed_yaml_file['Z_sol'],
                                    parsed_yaml_file['hZ'], lrad,
                                    brad, sptM, SWrad, parsed_yaml_file['SW'], parsed_yaml_file['hR'], nsolM)
            nW4L, nW4extL = mlt.NWN(Y_model, MYL, parsed_yaml_file['EBmV'], parsed_yaml_file['Z_sol'],
                                    parsed_yaml_file['hZ'], lrad,
                                    brad, sptL, SWrad, parsed_yaml_file['SW'], parsed_yaml_file['hR'], nsolL)

            nW4T, nW4extT = mlt.NWN(Y_model, MYT, parsed_yaml_file['EBmV'], parsed_yaml_file['Z_sol'],
                                    parsed_yaml_file['hZ'], lrad,
                                    brad, sptT, SWrad, parsed_yaml_file['SW'], parsed_yaml_file['hR'], nsolT)
        dnW4extM = mlt.dnWext(Y_model, nW4M, nW4extM, sptM)
        dnW4extL = mlt.dnWext(Y_model, nW4L, nW4extL, sptL)
        dnW4extT = mlt.dnWext(Y_model, nW4T, nW4extT, sptT)

        for n in range(nb_BD_M):
            #print(spts_M)
            P = mlt.PRIOR_M(n, Y_model, Yfirst, dnW4extM, spts_M)
            WS = mlt.WS(obj, n, Y_model, data_candidates, colors_M, ZP, Yfirst, num_filt, ref_filter, fluxlimite)
            Wsf_M[obj, n] = simps(P * WS[:], Y_model)

        for n in range(nb_BD_L):
            P = mlt.PRIOR_L(n, Y_model, Yfirst, dnW4extL, spts_L)
            WS = mlt.WS(obj, n, Y_model, data_candidates, colors_L, ZP, Yfirst, num_filt, ref_filter, fluxlimite)
            Wsf_L[obj, n] = simps(P * WS[:], Y_model)

        for n in range(nb_BD_T):
            P = mlt.PRIOR_T(n, Y_model, Yfirst, dnW4extT, spts_T)
            WS = mlt.WS(obj, n, Y_model, data_candidates, colors_T, ZP, Yfirst, num_filt, ref_filter, fluxlimite)
            Wsf_T[obj, n] = simps(P * WS[:], Y_model)

    start4 = time.time()
    print(f">>>wsJCL {start4 - start3}")
    #print(all_listsT)
    densL = np.zeros(nb_obj)
    densT = np.zeros(nb_obj)
    


    adjusted_all_listsL = []
    for l in all_listsL:
        start = l[0]
        end = l[1]
        if start <= 44 < end:  # Adjust if 44 is in this range
            if start == 44:  # if the start is 44, move start by 1
                start += 1
            end -= 1  # move the end by 1 since we removed index 44
        elif start > 44:  # if the start is after 44, move both start and end by 1
            start -= 1
            end -= 1
        adjusted_all_listsL.append([start, end])

# Replace the original list with the adjusted one
    all_listsL = adjusted_all_listsL
    Wsf_L = np.delete(Wsf_L, 44, axis=1)
    for n in range(10):
        nn = all_listsL[n]
        densL += (Wsf_L[:, nn[0]:nn[1]].mean(1))

    for n in range(9):
        print('n=',n)
        nn = all_listsT[
            n]


        print(nn)# stocker le premier indice et dernier indice dans la liste des types spectraux pour choisir une template aleatoirement
        densT += np.abs((Wsf_T[:, nn[0]:nn[1]].mean(1)))

    densM = Wsf_M.sum(1)
    import model.cont_gal as gal

    #gseds = 58

    Wgf = np.zeros((gseds, nb_obj))  # Marginal probability distribution for quasars

    zg = parsed_yaml_file["redshift_g"]
    redshiftg = np.arange(zg[0], zg[1], zg[2])
    cvg = cosmo.differential_comoving_volume(redshiftg).value * (np.pi / 180) ** 2
    colors_template_g = np.loadtxt(path_parent + colors_path + 'GLX_colors.txt')

    #### GALAXIES
    start5 = time.time()
    print(f">>>5 {start5 - start4}")

    maxi_densites, indices_maxi = np.zeros((gseds, len(data_candidates[0]))), np.zeros((gseds, len(data_candidates[0])))

    densites = np.zeros((gseds, len(data_candidates[0])))
    for sed in range(gseds):
        if ages[sed]<redshiftg.max():
            RG=np.linspace(redshiftg.min(),ages[sed],len(redshiftg))
        else:
            RG=redshiftg
        print('SEG gal= ', sed)
        M_G = np.loadtxt(path_parent + colors_path + arrg[sed]).T
        s = gal.galaxy_density(Y_model, RG, M_G, cvg)
        for obj in range(nb_obj):

            WG = qso.completude_likelihood(obj, Y_model, RG, colors_template_g,
                                           sed, ZP, data_candidates, num_filt, ref_filter, fluxlimite)
            Wgf[sed][obj] = simps(simps(s * WG, Y_model), RG)


            indices_maxi[sed, obj] = np.argmax((simps(s* WG, Y_model)))
            result = s*WG
            maxi_densites[sed, obj] = RG[int(indices_maxi[sed, obj])]
            densites[sed, obj] = simps(s*WG, Y_model)[int(indices_maxi[sed, obj])]
    max_indices = np.argmax(densites, axis=0)
    #indexstar = 0
    #maxi = (densites.T[indexstar]).max()
    Lg = np.zeros(nb_obj)
    
    for i in range(nb_obj):
        Lg[i] = maxi_densites[int(max_indices[i]), i]
    



    ws = (densT  + densL+densM)
    start6 = time.time()
    print(f">>>completude_likehood {start6 - start5}")
    # print(Wgf)

    # ws=Wsf_L.mean(1)+Wsf_T.mean(1)+Wsf_M.mean(1)
    pg = Wgf.mean(0) / (Wgf.mean(0) + ws + Wqf.mean(0))
    pq = Wqf.mean(0) / (Wgf.mean(0) + ws + Wqf.mean(0))
    ps = ws / (Wgf.mean(0) + ws + Wqf.mean(0))
    pm = densM / (Wgf.mean(0) + ws + Wqf.mean(0))
    pl = densL / (Wgf.mean(0) + ws + Wqf.mean(0))
    pt = densT / (Wgf.mean(0) + ws + Wqf.mean(0))


    # FIND which template gave the best fit


    def find_max_values_and_indices_per_column(Wgf):
        max_indices = np.argmax(Wgf, axis=0)
        return max_indices
    def find_max_values_and_indices_per_columnMLT(Wgf):
        max_indices = np.argmax(Wgf, axis=1)
        return max_indices

    indicesG = find_max_values_and_indices_per_column(Wgf)
    indicesM = find_max_values_and_indices_per_columnMLT(Wsf_M)
    indicesL = find_max_values_and_indices_per_columnMLT(Wsf_L)
    indicesT = find_max_values_and_indices_per_columnMLT(Wsf_T)
    indicesQ = find_max_values_and_indices_per_column(Wqf)

    #format_specifiers =['%.1f']*3+['%.4e']*12
    #format_specifiers = ['%.1f', '%.1f', '%.1f',  '%.4e', '%.4e', '%.4e', '%.4e',  '%.4e', '%.4e',  '%.4e', '%.4e', '%.4e', '%.4e', '%.4e', '%.4e','%.2f', '%.2e', '%.4e', '%.4e','%.4f','%2.f','%.4e', '%.4e', '%.4e', '%.4e','%.1f','%.1f','%.1f','%.1f','%.1f' ]
    file2 = path_parent + result_path + "R_" + cand_list[candi_index]
    combined_array = np.concatenate((data_candidates, np.array([ps, pg, pq, L,Wgf.mean(0),ws,Wqf.mean(0),pm,pl,pt,indicesM,indicesL,indicesT,indicesG,indicesQ,Lg])), axis=0)

    #header = "#ID  \t ra \t dec \t fvis \t svis \t  fy \t sy \t fj \t sj \t fh \t sh \t zin \t M1450 \t templatein \t ps \t pg \t pq \t zout \t wg \t  ws \t wq \t densM \t pm  \t pl \t pt \t tempM \t tempL \t tempT \t tempG \t tempQ \t zg \n"    # header="#ID \t ra  \t dec \t fz \t errz \t fy \t sy \t fj \t sj \t fh \t sh \t fk \t  errk \t  fw1 \t  errw1 \t  fw2 \t  fw2 \t  zin  \t \ps \t pg \t pq \t zout \n "
    # Combine header and data

    header = "#ID  \t l \t b \t fvis \t svis \t  fy \t sy \t fj \t sj \t fh \t sh  \t  ps \t pg \t pq \t zout \t wg \t  ws \t wq  \t pm  \t pl \t pt \t tempM \t tempL \t tempT \t tempG \t tempQ \t zg \n"    # header="#ID \t ra  \t dec \t fz \t errz \t fy \t sy \t fj \t sj \t fh \t sh \t fk \t  errk \t  fw1 \t  errw1 \t  fw2 \t  fw2 \t  zin  \t \ps \t pg \t pq \t zout \n "


    # File path

    # Write header and data to file
    with open(file2, 'w') as f:
        f.write(header)
        np.savetxt(f, combined_array.T, delimiter='\t')

    # Measure time
    end = time.time()
  
