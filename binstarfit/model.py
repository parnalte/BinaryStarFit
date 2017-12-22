#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:27:59 2017

@author: pablo

Contains the functions defining our model for the orbits.

Basically, taken directly from Wright&Howard (2009)
"""

import numpy as np
import scipy.optimize as so

RAD2DEG = 180./np.pi
DEG2RAD = 1./RAD2DEG


def _get_ThieleInnes_from_params(a, angle_O_deg, angle_w_deg, angle_i_deg):
    """
    Compute the Thiele-Innes parameters A, B, F, G needed for the orbit 
    calculation from the primary star's orbit parameters.
    
    INPUT:
        a: semi-major axis of the primary star's orbit [arcsec]
        angle_O_deg: position angle of the ascending node [degrees]
        angle_w_deg: argument of periastron of the primary star [degrees]
        angle_i_deg: inclination of the orbit with respect to the sky [degrees]
        
    OUTPUT:
        A, B, F, G: corresponding Thiele-Innes parameters for the orbit
    """
    
    sin_O = np.sin(angle_O_deg*DEG2RAD)
    cos_O = np.cos(angle_O_deg*DEG2RAD)
    sin_w = np.sin(angle_w_deg*DEG2RAD)
    cos_w = np.cos(angle_w_deg*DEG2RAD)
    cos_i = np.cos(angle_i_deg*DEG2RAD)
    
    TIpar_A = a*( (cos_O*cos_w) - (sin_O*sin_w*cos_i) )
    TIpar_B = a*( (sin_O*cos_w) + (cos_O*sin_w*cos_i) )
    TIpar_F = a*( (-cos_O*sin_w) - (sin_O*cos_w*cos_i) )
    TIpar_G = a*( (-sin_O*sin_w) + (cos_O*cos_w*cos_i) )
    
    return TIpar_A, TIpar_B, TIpar_F, TIpar_G


def _f_Kepler(x, eccentricity, M):
    """
    Auxiliary function for solving Kepler's equation.
    It is defined such that f_Kepler(x=E) = 0, where E is the eccentric
    anomaly.
    
    INPUT: 
        x: argument of function
        eccentricity: idem of the orbit
        M: mean anomaly, defined as 2 pi (t - t0) / P
        
    OUTPUT:
        f(x)
    """
    return x - eccentricity*np.sin(x) - M

def _get_ecc_anomaly(tval, eccentricity, period, T0):
    """
    Function to compute the eccentric anomaly corresponding to a particular
    time, given the needed orbit parameters.
    The eccentric anomaly is obtained by solving Kepler's Equation using the 
    classical Brent method (see Scipy's help for details).
    
    INPUT:
        tval: time [years, or other consistent units]
        eccentricity: eccentricity of orbit
        period: period of the orbit [years, or other consistent units]
        T0: time of periastron passage [years, or other consistent units]
        
    OUTPUT:
        eccentric anomaly for this time given the orbit parameters
    """
    
    cycle_no = np.floor( (tval - T0) / period)
    low_lim = 2.*np.pi*cycle_no
    upp_lim = 2.*np.pi*(cycle_no + 1)
    M_t = 2.*np.pi*(tval - T0) / period
    
    return so.brentq(f=_f_Kepler, args=(eccentricity, M_t),
                     a=low_lim, b=upp_lim)


def _get_XY_from_params(eccentricity, ecc_anomaly):
    """
    Function to compute the X,Y auxiliary parameters (eqs. 51, 52 in
    Wright&Howard) from the eccentricity and eccentric anomaly.
    
    INPUT:
        eccentricity
        ecc_anomaly
        
    OUTPUT:
        X, Y: auxiliary parameters needed for the orbit calculation
    """
    
    X = np.cos(ecc_anomaly) - eccentricity
    Y = np.sqrt(1 - (eccentricity**2))*np.sin(ecc_anomaly)
    return X, Y



def _get_XY_from_time(tval, eccentricity, period, T0):
    """
    Wrapper function to obtain the X,Y auxiliary parameters (eqs. 51, 52 in
    Wright&Howard) directly from the time and orbit parameters.
    
    INPUT:
        tval: time [years, or other consistent units]
        eccentricity: eccentricity of orbit
        period: period of the orbit [years, or other consistent units]
        T0: time of periastron passage [years, or other consistent units]

    OUTPUT:
        X, Y: auxiliary parameters needed for the orbit calculation
    """
    ecc_anom = _get_ecc_anomaly(tval, eccentricity, period, T0)
    return _get_XY_from_params(eccentricity, ecc_anom)


def compute_orbit_primary(tvals, angle_O_deg, angle_w_deg, angle_i_deg, a_axis,
                        eccentricity, period, T0):
    """
    Function to compute the astrometric displacements for the primary due to
    the orbit, for several time values, given the orbit's parameters.
    
    The results I compute here are the displacements, in the DEC and RA
    directions, with respect to the centre of mass of the system, in the
    appropriate projected rectilinear coordinate system (i.e. that used in
    Wright&Howard).
    
    INPUT:
        tvals: times, either single value or array [years]
        angle_O_deg: position angle of the ascending node [degrees]
        angle_w_deg: argument of periastron of the primary star [degrees]
        angle_i_deg: inclination of the orbit with respect to the sky [degrees]
        a_axis: semi-major axis of the primary star's orbit [arcsec]
        eccentricity: eccentricity of orbit
        period: period of the orbit [years, or other consistent units]
        T0: time of periastron passage [years, or other consistent units]

    OUTPUT:
        orbit_displ_dec: astrometric displacement of the primary star with
            respect to the CoM in the declination direction [arcsec]
        orbit_displ_ra: astrometric displacement of the primary star with
            respect to the CoM in the right ascention direction, in the
            appropriate rectilinear coordinates (i.e. taking into account the
            cos(delta) term using the nominal delta value)  [arcsec]
    """
    
    tvals = np.atleast_1d(tvals)
    Nt = len(tvals)
    orbit_displ_dec = np.empty(Nt)
    orbit_displ_ra = np.empty(Nt)
    
    A, B, F, G = _get_ThieleInnes_from_params(a_axis, angle_O_deg, angle_w_deg,
                                             angle_i_deg)
    
    for i,t in enumerate(tvals):
        X, Y = _get_XY_from_time(t, eccentricity, period, T0)
        orbit_displ_dec[i] = A*X + F*Y
        orbit_displ_ra[i] = B*X + G*Y
        
    return orbit_displ_dec, orbit_displ_ra