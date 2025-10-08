"""
-----------------------------------------------------
Author : Meriam Ezziati
Date   : 8 April 2022
------------------------------------------------------

MODULE MLT: Functions to calculate the marginal distribution for the bayesian
selection method for MLT stars

"""
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
import model.qso as qso


def distance(m: float, mabs: float):
    """
    This function calculates distance from magnitudes
    :param float m: apparent magnitude
    :param float mabs: absolute magnitude
    :return: the distance
    """
    return 10 ** ((m - mabs + 5) / 5.)


def surface(dim: float):
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


def I(d: float, dB: float):
    """
    Equation 4.22 that allows us through an integration to calculate the spacial density per spectral type
    :param d: heliocentric distance
    :param dB: heliocentric distance only dependent on galactic coordinates

    """
    return (-1.) * np.exp((-d) / dB) * (dB * d ** 2 + 2 * dB ** 2 * d + 2 * dB ** 3)


def Nnord(d, l, b, S, nsol, z_sol, hZ, hR):  # NGP
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
    n0 = nsol / np.exp(-z_sol / hZ)
    nA = n0 * np.exp(-z_sol / hZ)
    dBinvert = (-np.cos(b) * np.cos(l)) / hR + np.sin(b) / hZ
    dB = 1. / dBinvert
    Nb = S * (nA * (I(d, dB) - I(0, dB)))
    return Nb



def Nsud(d, l, b, S, nsol, z_sol, hZ, hR):
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
    n0 = nsol/np.exp(-z_sol/hZ)
    dstar = -z_sol/np.sin(b)

    nAplus = n0*np.exp(-z_sol/hZ)
    dBinvertplus = (-np.cos(b)*np.cos(l))/hR + np.sin(b)/hZ
    dBplus = 1./dBinvertplus

    nAmoins = n0*np.exp(z_sol/hZ)
    dBinvertmoins = (-np.cos(b)*np.cos(l))/hR - np.sin(b)/hZ
    dBmoins = 1./dBinvertmoins
    Nb = S*(nAplus*(I(dstar,dBplus)-I(0.,dBplus))+nAmoins*(I(d,dBmoins)-I(dstar,dBmoins)))

    return Nb


def NWS(Y, MY, EBmV, Z_sol, hZ, lrad, brad, spt, SWrad, SW, hR, nsol):
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

    NW = np.zeros((len(spt), len(Y)))
    nW = np.zeros((len(spt), len(Y)))
    NWext = np.zeros((len(spt), len(Y)))
    nWext = np.zeros((len(spt), len(Y)))
    for i in range(len(spt)):
        for j in range(len(Y)):
            dist = distance(Y[j], MY[i])
            distext = float(fsolve(FW, x0=dist, args=(Y[j], MY[i], EBmV, Z_sol, hZ, brad)))
            NW[i, j] = Nsud(dist, lrad, brad, SWrad, nsol[i], Z_sol, hZ, hR)
            nW[i, j] = NW[i, j] / (SW)
            NWext[i, j] = Nsud(distext, lrad, brad, SWrad, nsol[i], Z_sol, hZ, hR)
            nWext[i, j] = NWext[i, j] / (SW)

    return nW, nWext

def NWN(Y, MY, EBmV, Z_sol, hZ, lrad, brad, spt, SWrad, SW, hR, nsol):
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

    NW = np.zeros((len(spt), len(Y)))
    nW = np.zeros((len(spt), len(Y)))
    NWext = np.zeros((len(spt), len(Y)))
    nWext = np.zeros((len(spt), len(Y)))
    for i in range(len(spt)):
        for j in range(len(Y)):
            dist = distance(Y[j], MY[i])
            distext = float(fsolve(FW, x0=dist, args=(Y[j], MY[i], EBmV, Z_sol, hZ, brad)))
            NW[i, j] = Nnord(dist, lrad, brad, SWrad, nsol[i], Z_sol, hZ, hR)
            nW[i, j] = NW[i, j] / (SW)
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


def rho_bd_ext(Yfirst, magY, spt, dnW4ext):
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
    liste = ["M3", "M4", "M5", "M6", "M7", "M8", "M9", "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "T0",
             "T1", "T2", "T3", "T4", "T5", "T6", "T7",
             "T8"]
    indice = liste.index(str(spt[:2]))

    f = interpolate.interp1d(Yfirst, dnW4ext[indice], fill_value="extrapolate")

    return f(magY)


def ws(k, n, magH, data, colors, zp, Hfirst, dnW4ext, sty,num_filt,ref_filter):
    """
    Marginal distribution  of probability
    :param k: index object
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
    gauss_i = np.zeros((num_filt, len(data[0]), len(colors[0]), len(magH)))
    I1 = np.zeros((len(data[0]), len(colors[0]), len(magH)))
    I2 = np.zeros((len(data[0]), len(colors[0]), len(magH)))
    stype = sty[n][:2]

    I1[k, n, :] = (rho_bd_ext(Hfirst, magH[:], stype, dnW4ext))
    I2[k, n, :] = qso.completude(magH[:])

    for f in range(num_filt):
        if f == ref_filter:
            magi = magH
            fluxi = qso.flux(magi, zp)
            gauss_i[f, k, n, :] = abs(qso.flux_int(magi, zp)) * qso.gauss(data[2 * f + 1+2][k], fluxi, data[2 * f + 2+2][k])
        else:
            if(f<ref_filter):
                nn=f
            else:
                nn=f-1
            h_vis = colors[nn]
            magi = -1.0 * h_vis[n] + magH[:]
            fluxi = qso.flux(magi, zp)
            gauss_i[f,k, n, :] = abs(qso.flux_int(magi, zp)) * qso.gauss(data[2 * f + 1+2][k], fluxi, data[2 * f + 2+2][k])

    I3 = np.prod(gauss_i, axis=0)


    return I1, I2, I3


def stardensity(k, n, magH, data, colors, Hfirst, dnW4ext, sty):
    """
    Marginal distribution  of probability
    :param k: index object
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
    I1 = np.zeros((len(data[0]), len(colors[0]), len(magH)))
    stype = sty[n][:2]

    I1[k, n, :] = (rho_bd_ext(Hfirst, magH[:], stype, dnW4ext))
    return I1

def wsJCL(k, n, magH, data, colors, zp, Hfirst, dnW4ext, sty,num_filt,ref_filter):
    """
    Marginal distribution  of probability
    :param k: index object
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
    I1 = np.zeros(len(magH))
    #I2 = np.zeros(len(magH))
    stype = sty[n][:2]

    I1 = (rho_bd_ext(Hfirst, magH[:], stype, dnW4ext))
    #I2 = qso.completude(magH[:])

    for f in range(num_filt):
        if f == ref_filter:
            magi = magH
            fluxi = qso.flux(magi, zp)
            gauss_i[f,:] = abs(qso.flux_int(magi, zp)) * qso.gauss(data[2 * f + 1+2][k], fluxi, data[2 * f + 2+2][k])
        else:
            if(f<ref_filter):
                nn=f
            else:
                nn=f-1
            h_vis = colors[nn]
            magi = -1.0 * h_vis[n] + magH[:]
            fluxi = qso.flux(magi, zp)
            gauss_i[f,:] = abs(qso.flux_int(magi, zp)) * qso.gauss(data[2 * f + 1+2][k], fluxi, data[2 * f + 2+2][k])

    I3 = np.prod(gauss_i, axis=0)


    return I1, I3

def wsJC( n, magH, Hfirst, dnW4ext, sty):
    """
    Marginal distribution  of probability
    :param k: index object
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

    I1 = (rho_bd_ext(Hfirst, magH[:], stype, dnW4ext))

    return I1
