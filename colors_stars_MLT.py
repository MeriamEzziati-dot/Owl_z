import numpy as np
from colors.LT_colors  import *
import time
import os
import matplotlib.pyplot as plt
import yaml


a_yaml_file = open("config.yaml")

if __name__ == "__main__":
    parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    c = float(parsed_yaml_file["light_speed"])
    const = float(parsed_yaml_file["const"])
    start = time.time()
    # Path management
    path_parent = str(parsed_yaml_file['workdir'])
    template_path_lt = str(parsed_yaml_file['templates_MLT'])
    filter_path = str(parsed_yaml_file['filters_dir'])
    colors_path=str(parsed_yaml_file['colors_result_path'])
    filters_list = str(parsed_yaml_file['filters_list'])

    filters = np.genfromtxt(path_parent + filters_list, dtype=str)
    ref_filter = parsed_yaml_file["reference_filt"]
    num_filt = len(filters)
    limag=parsed_yaml_file["LIM_mag"]
    ref = np.genfromtxt(path_parent + filter_path + filters[int(ref_filter) ]).T



    templates_L_list = parsed_yaml_file['templates_L_list']
    templates_M_list = parsed_yaml_file['templates_M_list']
    templates_T_list = parsed_yaml_file['templates_T_list']
###################################################################
                     # M
###################################################################

    arrM = np.genfromtxt(path_parent + templates_M_list, dtype=str)
    arrL = np.genfromtxt(path_parent + templates_L_list, dtype=str)
    arrT = np.genfromtxt(path_parent + templates_T_list, dtype=str)
    print(path_parent + template_path_lt+arrT[51])

    #arr=np.delete(arr,list(arr).index('L4_7.txt'))
    fileM = np.zeros((int(num_filt)-1,len(arrM)))

    for i in range( len(arrM)):
        k = 0
        for f in range(num_filt):
            if f == ref_filter :
                continue
            else:
                filter = np.genfromtxt(path_parent + filter_path + filters[f]).T
                data = np.loadtxt(path_parent+template_path_lt + arrM[i]).T
                mref = colors_lt(data, const, ref,c)
                mobs = colors_lt(data, const, filter, c)
                if mobs==99:
                    fileM[k][i]=999
                else:
                    fileM[k][i] = mref - mobs
                k += 1
    np.savetxt(path_parent+colors_path +"M_colors.txt", fileM.T,  header="Y-J,J-H")

    #np.savetxt(result_path + 'MLT_colors2.dat', file.T)
    end=time.time()
    print('Finished calculating M colors in', end-start, ' seconds')
###################################################################
                     # L
###################################################################


    #arr=np.delete(arr,list(arr).index('L4_7.txt'))
    fileL = np.zeros((int(num_filt)-1,len(arrL)))

    for i in range( len(arrL)):
        data = np.loadtxt(path_parent + template_path_lt + arrL[i]).T
        k = 0
        for f in range(num_filt):
            if f == ref_filter :
                continue
            else:
                filter = np.genfromtxt(path_parent + filter_path + filters[f]).T
                mref = colors_lt(data, const, ref, c)
                mobs = colors_lt(data, const, filter, c)
                if mobs ==99:
                    fileL[k][i] = 999
                else:
                    fileL[k][i] = mref - mobs
                k += 1
    np.savetxt(path_parent+colors_path +"L_colors.txt", fileL.T,  header="Y-J,J-H")


    #np.savetxt(result_path + 'MLT_colors2.dat', file.T)
    end=time.time()
    print('Finished calculating L colors in', end-start, ' seconds')

###################################################################
                     # T
###################################################################

    arrT = np.genfromtxt(path_parent + templates_T_list, dtype=str)
    #arr=np.delete(arr,list(arr).index('L4_7.txt'))
    fileT = np.zeros((int(num_filt)-1,len(arrT)))

    for i in range( len(arrT)):
        k = 0
        for f in range(num_filt):
            if f == ref_filter :
                continue
            else:
                filter = np.genfromtxt(path_parent + filter_path + filters[f]).T
                data = np.loadtxt(path_parent + template_path_lt + arrT[i]).T
                mref = colors_lt(data, const, ref, c)
                mobs = colors_lt(data, const, filter, c)
                if mobs ==99:
                    fileT[k][i] = 999
                else:
                    fileT[k][i] = mref - mobs
                k += 1
    np.savetxt(path_parent+colors_path +"T_colors.txt", fileT.T,  header="Y-J,J-H")

    end=time.time()
    print('Finished calculating T colors in', end-start, ' seconds')