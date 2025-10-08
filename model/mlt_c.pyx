"""
-----------------------------------------------------
Author : Meriam Ezziati
Date   : 8 April 2022
------------------------------------------------------
Cythonized by Jean-Charles LAMBERT - March 2024

MODULE MLT: Functions to calculate the marginal distribution for the bayesian
selection method for MLT stars

"""
from scipy import special, interpolate

cimport cython
import numpy as np
from cython.parallel cimport prange,parallel
cimport openmp
from openmp cimport omp_get_thread_num,omp_get_num_threads


#import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
#import model.qso as qso
import model.qso_c as qso

def distance(double m, double mabs):
    """
    This function calculates distance from magnitudes
    :param float m: apparent magnitude
    :param float mabs: absolute magnitude
    :return: the distance
    """
    return 10 ** ((m - mabs + 5) / 5.)


def surface(double dim):
    """
    Return Surface corresponding to the given distance
    :param dim: dimension to compute surface from
    :return:
    """
    return dim * ((np.pi / 180.) ** 2)


def d_star(b: np.ndarray, z_sol: float, hZ: float):
    """
    This function compute the distance observer (sun) to edge of thin disk
    :param np.ndarray sn: South or north of galactic plane array
    :param b: galactic latitude coordinate
    :param z_sol: height of the solar system  above the galactic plane
    :param hZ:radian length scale
    :return:
    """

    dstar = (hZ * 0.5 - z_sol) / np.sin(b)
    return dstar


def FW(d: float, magY: float, MabsY: float, EBmV: np.ndarray, z_sol: float, hZ: float, brad: np.ndarray):
    """
    equation 4.21  Sarah' manuscript
    to  calculate the heliocentric distance  as a function of t the apparent magnitude Y(magY)
    :param d: heliocentric distance
    :param magY: apparent magnitude
    :param MabsY: absolute magnitude
    :param EBmV: mean extinction
    :param z_sol: height of the solar system  above the galactic plane
    :param hZ: radian length scale
    :param brad: galactic latitude in radians
    :param field: observed field W1, W2,W3 or W4
    :param sn: north or south galactic plane?
    """
    Az_over_Av = 0.39
    E = EBmV
    dstar = d_star(brad, z_sol, hZ)
    Az = 3.09 * E * d / dstar * Az_over_Av
    return 5 * np.log10(d) - 5 + Az - (magY - MabsY)


def I(double d, double dB):
    """
    Equation 4.22 that allows us through an integration to calculate the spacial density per spectral type
    :param d: heliocentric distance
    :param dB: heliocentric distance only dependent on galactic coordinates

    """
    return (-1.) * np.exp((-d) / dB) * (dB * d ** 2 + 2 * dB ** 2 * d + 2 * dB ** 3)


def Nnord(double d, double l, double b, double S, double nsol, double z_sol, double hZ, double hR):  # NGP
    """
    the spacial density per spectral type
    :param d:  heliocentric distancemagY
    :param l:  galactic latitude
    :param b:  galactic longitude
    :param S: surface
    :param nsol: local density
    :param z_sol: height of the solar system  above the galactic plane
    :param hZ: radian length scale
    :param hR: height of the galactic disk
    :return:  number of brown dwarfs per spectral type above the galactic plane
    """
    cdef double n0 = nsol / np.exp(-z_sol / hZ)
    cdef double nA = n0 * np.exp(-z_sol / hZ)
    cdef double dBinvert = (-np.cos(b) * np.cos(l)) / hR + np.sin(b) / hZ
    cdef double dB = 1. / dBinvert
    cdef double Nb = S * (nA * (I(d, dB) - I(0, dB)))
    return Nb



def Nsud(double d, double l, double b, double S, double nsol, double z_sol, double hZ, double hR):
    """
    the spacial density per spectral type
    :param d:  heliocentric distancemagY
    :param l:  galactic latitude
    :param b:  galactic longitude
    :param S: surface
    :param nsol: local density
    :param z_sol: height of the solar system  above the galactic plane
    :param hZ: radian length scale
    :param hR: height of the galactic disk
    :return:  number of brown dwarfs per spectral type above the galactic plane
    """
    cdef double n0 = nsol/np.exp(-z_sol/hZ)
    cdef double dstar = -z_sol/np.sin(b)

    cdef double nAplus = n0*np.exp(-z_sol/hZ)
    cdef double dBinvertplus = (-np.cos(b)*np.cos(l))/hR + np.sin(b)/hZ
    cdef double dBplus = 1./dBinvertplus

    cdef double nAmoins = n0*np.exp(z_sol/hZ)
    cdef double dBinvertmoins = (-np.cos(b)*np.cos(l))/hR - np.sin(b)/hZ
    cdef double dBmoins = 1./dBinvertmoins
    cdef double Nb = S*(nAplus*(I(dstar,dBplus)-I(0.,dBplus))+nAmoins*(I(d,dBmoins)-I(dstar,dBmoins)))

    return Nb

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def NWS(double [:] Y, double [:] MY, double EBmV, double Z_sol, double hZ, double lrad, double brad, spt, double SWrad,
        double  SW, double hR, double [:] nsol):
    """
    Compute  N / density for all fields
    :param field: field W1,W2,W3 or W4
    :param Y: apparent magnitude Y
    :param MY: Absolute magnitude Y
    :param EBmV: mean extinction
    :param nsol: local density
    :param Z_sol: height of the solar system  above the galactic plane
    :param hZ: radian length scale
    :param lrad: galactic longitude in rad
    :param brad: galactic latitude in rad
    :param sn: South or north galactic plane
    :param spt: spectral type
    :param SWrad: surface in rad^2
    :param SW: surface
    :param hR: height of the galactic disk
    :param nsol: local density
    """

    cdef int i,j,spt_l,Y_l
    spt_l = len(spt)
    Y_l = len(Y)

    cdef double [:,:] NW = np.zeros((spt_l,Y_l))
    cdef double [:,:] nW = np.zeros((spt_l,Y_l))
    cdef double [:,:] NWext = np.zeros((spt_l,Y_l))
    cdef double [:,:] nWext = np.zeros((spt_l,Y_l))

    cdef double dist, distext
    
    for i in range(spt_l):
        for j in range(Y_l):
            dist = distance(Y[j], MY[i])
            distext = float(fsolve(FW, x0=dist, args=(Y[j], MY[i], EBmV, Z_sol, hZ, brad)))
            NW[i, j] = Nsud(dist, lrad, brad, SWrad, nsol[i], Z_sol, hZ, hR)
            nW[i, j] = NW[i, j] / (SW)
            NWext[i, j] = Nsud(distext, lrad, brad, SWrad, nsol[i], Z_sol, hZ, hR)
            nWext[i, j] = NWext[i, j] / (SW)

    return nW, nWext

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def NWN(double [:] Y, double [:] MY, double EBmV, double Z_sol, double hZ, double lrad, double brad, spt, double SWrad,
        double SW, double hR, double [:] nsol):
    """
    Compute  N / density for all fields
    :param field: field W1,W2,W3 or W4
    :param Y: apparent magnitude Y
    :param MY: Absolute magnitude Y
    :param EBmV: mean extinction
    :param nsol: local density
    :param Z_sol: height of the solar system  above the galactic plane
    :param hZ: radian length scale
    :param lrad: galactic longitude in rad
    :param brad: galactic latitude in rad
    :param sn: South or north galactic plane
    :param spt: spectral type
    :param SWrad: surface in rad^2
    :param SW: surface
    :param hR: height of the galactic disk
    :param nsol: local density
    """

    cdef int i,j,spt_l,Y_l
    spt_l = len(spt)
    Y_l = len(Y)

    cdef double [:,:] NW = np.zeros((spt_l,Y_l))
    cdef double [:,:] nW = np.zeros((spt_l,Y_l))
    cdef double [:,:] NWext = np.zeros((spt_l,Y_l))
    cdef double [:,:] nWext = np.zeros((spt_l,Y_l))

    cdef double dist, distext

    for i in range(spt_l):
        for j in range(Y_l):
            dist = distance(Y[j], MY[i])
            distext = float(fsolve(FW, x0=dist, args=(Y[j], MY[i], EBmV, Z_sol, hZ, brad)))
            NWext[i, j] = Nnord(distext, lrad, brad, SWrad, nsol[i], Z_sol, hZ, hR)
            nWext[i, j] = NWext[i, j] / (SW)

    return nW, nWext


def dnWext(Y, nW, nWext, spt):
    """
    First derivative of N / density to express it in deg^-2.mag^-1
    :param Y: apparent magnitude Y
    :param nW: N / density
    :param nWext: N / density/surface
    :param spt: spectral type
    """
    dnW1 = []

    dx = np.diff(Y)

    dnW1ext = []

    for i in range(len(spt)):
        dyW1 = np.diff(nW[i])
        dnW1.append(dyW1 / dx)

        dyW1ext = np.diff(nWext[i])
        dnW1ext.append(dyW1ext / dx)

    return dnW1ext

def rho_bd_ext_M(Yfirst, magY, spt, dnW4ext):
    """
    :param Yfirst:
    :param magY: apparent magnitude Y
    :param spt: spectral type
    :param champ: fields W1,W2,W3 or W4
    :param dnW1ext: N / density/surface W1
    :param dnW2ext: N / density/surface W2
    :param dnW3ext: N / density/surface W3
    :param dnW4ext: N / density/surface W4
    """
    #liste = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7",
    #         "T8"]
    liste = ["M3", "M4", "M5", "M6", "M7", "M8", "M9"]
    indice = liste.index(str(spt[:2]))
    f = interpolate.interp1d(Yfirst, dnW4ext[indice], fill_value="extrapolate")

    return f(magY)
def rho_bd_ext_L(Yfirst, magY, spt, dnW4ext):
    """
    :param Yfirst:
    :param magY: apparent magnitude Y
    :param spt: spectral type
    :param champ: fields W1,W2,W3 or W4
    :param dnW1ext: N / density/surface W1
    :param dnW2ext: N / density/surface W2
    :param dnW3ext: N / density/surface W3
    :param dnW4ext: N / density/surface W4
    """
    #liste = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7",
    #         "T8"]
    liste = [ "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
    indice = liste.index(str(spt[:2]))
    f = interpolate.interp1d(Yfirst, dnW4ext[indice], fill_value="extrapolate")

    return f(magY)
def rho_bd_ext_T(Yfirst, magY, spt, dnW4ext):
    """
    :param Yfirst:
    :param magY: apparent magnitude Y
    :param spt: spectral type
    :param champ: fields W1,W2,W3 or W4
    :param dnW1ext: N / density/surface W1
    :param dnW2ext: N / density/surface W2
    :param dnW3ext: N / density/surface W3
    :param dnW4ext: N / density/surface W4
    """
    #liste = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7",
    #         "T8"]
    liste = [ "T0",
             "T1", "T2", "T3", "T4", "T5", "T6", "T7",
             "T8"]
    indice = liste.index(str(spt[:2]))
    f = interpolate.interp1d(Yfirst, dnW4ext[indice], fill_value="extrapolate")

    return f(magY)


def gaussian_cdf(x,mu,sigma):
    return 0.5 * (1 + special.erf((x - mu) / (sigma * np.sqrt(2))))



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing

def WS(int k, int n, magH, double [:,:] data, double [:,:] colors, double zp, double [:] Hfirst,int num_filt, int ref_filter,fluxlimite):
    """
    Marginal distribution  of probability
    :param k: index kect
    :param n: index spectral type
    :param magY: apparent magnitude
    :param data: candidates data
    :param colors: color data
    :param zp: Zero point
    :param Yfirst:
    :param dnW1ext: N / density/surface W1
    :param dnW2ext: N / density/surface W2
    :param dnW3ext: N / density/surface W3
    :param dnW4ext: N / density/surface W4
    :return:
    """
    gauss_i = np.zeros((num_filt,  len(magH)))
    l_magH=len(magH)

    cdef Py_ssize_t f
    cdef int nn


    for f in range(num_filt):
        F = data[2 * f + 1 + 2][k]

        errf = data[2 * f + 4][k]

        if f == ref_filter:
            f_int2 = -0.9210340371976183 * qso.flux(magH, zp)
            #LAM = 1 / (1 - qso.gaussian_cdf(0, qso.flux(magH, zp), errf))
            LAM=1.
            gauss_i[f, :] = LAM * abs(f_int2) * qso.gauss(F, qso.flux(magH, zp), errf)


        else:

            if (f < ref_filter):
                nn = f
            else:
                nn = f - 1

            h_vis = colors[nn]
            magi = (-1.0 * h_vis[n]) + magH[:]

            if (h_vis[n] == 999):
                fluxi =1e-60
            else:
                fluxi = qso.flux(magi, zp)
            LAM=1.
            #LAM = 1 / (1 - gaussian_cdf(0, fluxi, errf))


            gauss_i[f, :] = LAM * qso.gauss(F, fluxi, errf)



    I3 = np.prod(gauss_i, axis=0)
    return  I3




@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing

def PRIOR_M( int n, magH, double [:] Hfirst, dnW4ext,  sty):
    """
    Marginal distribution  of probability
    :param k: index kect
    :param n: index spectral type
    :param magY: apparent magnitude
    :param data: candidates data
    :param colors: color data
    :param zp: Zero point
    :param Yfirst:
    :param dnW1ext: N / density/surface W1
    :param dnW2ext: N / density/surface W2
    :param dnW3ext: N / density/surface W3
    :param dnW4ext: N / density/surface W4
    :return:
    """

    stype = sty[n][:2]
    I1 = (rho_bd_ext_M(Hfirst, magH[:], stype, dnW4ext))
    return I1


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def PRIOR_L( int n, magH, double [:] Hfirst, dnW4ext,  sty):
    """
    Marginal distribution  of probability
    :param k: index kect
    :param n: index spectral type
    :param magY: apparent magnitude
    :param data: candidates data
    :param colors: color data
    :param zp: Zero point
    :param Yfirst:
    :param dnW1ext: N / density/surface W1
    :param dnW2ext: N / density/surface W2
    :param dnW3ext: N / density/surface W3
    :param dnW4ext: N / density/surface W4
    :return:
    """

    stype = sty[n][:2]
    I1 = (rho_bd_ext_L(Hfirst, magH[:], stype, dnW4ext))
    return I1
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def PRIOR_T( int n, magH, double [:] Hfirst, dnW4ext,  sty):
    """
    Marginal distribution  of probability
    :param k: index kect
    :param n: index spectral type
    :param magY: apparent magnitude
    :param data: candidates data
    :param colors: color data
    :param zp: Zero point
    :param Yfirst:
    :param dnW1ext: N / density/surface W1
    :param dnW2ext: N / density/surface W2
    :param dnW3ext: N / density/surface W3
    :param dnW4ext: N / density/surface W4
    :return:
    """

    stype = sty[n][:2]
    I1 = (rho_bd_ext_T(Hfirst, magH[:], stype, dnW4ext))

    return I1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing

def WS_old(int k, int n, magH, double [:,:] data, double [:,:] colors, double zp, double [:] Hfirst,int num_filt, int ref_filter,fluxlimite):
    """
    Marginal distribution  of probability
    :param k: index kect
    :param n: index spectral type
    :param magY: apparent magnitude
    :param data: candidates data
    :param colors: color data
    :param zp: Zero point
    :param Yfirst:
    :param dnW1ext: N / density/surface W1
    :param dnW2ext: N / density/surface W2
    :param dnW3ext: N / density/surface W3
    :param dnW4ext: N / density/surface W4
    :return:
    """
    gauss_i = np.zeros((num_filt,  len(magH)))
    l_magH=len(magH)

    cdef Py_ssize_t f
    cdef int nn


    for f in range(num_filt):
        F = data[2 * f + 1 + 2][k]
        
        errf = data[2 * f + 4][k]
        SNR=F/errf
        if f == ref_filter:
            f_int2 = -0.9210340371976183 * qso.flux(magH, zp)
            LAM = 1 / (1 - qso.gaussian_cdf(0, qso.flux(magH, zp), errf))

            gauss_i[f, :] = LAM * abs(f_int2) * qso.gauss(F, qso.flux(magH, zp), errf)

        else:

            if (f < ref_filter):
                nn = f
            else:
                nn = f - 1

            h_vis = colors[nn]
            magi = (-1.0 * h_vis[n]) + magH[:]

            if (h_vis[n] == 999):
                fluxi = 1e-60
            else:
                fluxi = qso.flux(magi, zp)
            LAM = 1 / (1 - gaussian_cdf(0, fluxi, errf))

            if SNR<1:
                gauss_i[f, :] = LAM * qso.gaussian_cdf(fluxlimite[f] * np.ones(l_magH), fluxi, errf)

            else:
                gauss_i[f, :] = LAM * qso.gauss(F, fluxi, errf)



    I3 = np.prod(gauss_i, axis=0)
    return  I3

