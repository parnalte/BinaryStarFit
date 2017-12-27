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

import astropy.units as u
import astropy.coordinates as coord
from astropy.time import Time

RAD2DEG = 180./np.pi
DEG2RAD = 1./RAD2DEG


# FUNCTIONS NEEDED FOR THE COMPUTATION OF THE ORBIT OF THE PRIMARY STAR

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



# FUNCTIONS NEEDED FOR THE COMPUTATION OF THE ORBIT OF THE SECONDARY STAR,
# AND THE RELATIVE ORBIT

def _get_ThieleInnes_from_params_secondary(a_prim, angle_O_deg,
                                           angle_w_deg_prim, angle_i_deg,
                                           q_mass):
    """
    Compute the Thiele-Innes parameters A, B, F, G needed for the orbit 
    calculation of the secondary star from the *primary star's* orbit
    parameters, and the mass ratio.
    
    INPUT:
        a_prim: semi-major axis of the primary star's orbit [arcsec]
        angle_O_deg: position angle of the ascending node [degrees]
        angle_w_deg_prim: argument of periastron of the primary star [degrees]
        angle_i_deg: inclination of the orbit with respect to the sky [degrees]
        q_mass: mass ratio of the two stars, m_secondary/m_primary
        
    OUTPUT:
        A, B, F, G: corresponding Thiele-Innes parameters for the orbit of the
            secondary star
    """
    
    a = a_prim/q_mass
    
    sin_O = np.sin(angle_O_deg*DEG2RAD)
    cos_O = np.cos(angle_O_deg*DEG2RAD)
    sin_w = -np.sin(angle_w_deg_prim*DEG2RAD) # w_sec = w_prim + pi
    cos_w = -np.cos(angle_w_deg_prim*DEG2RAD) # w_sec = w_prim + pi
    cos_i = np.cos(angle_i_deg*DEG2RAD)
    
    TIpar_A = a*( (cos_O*cos_w) - (sin_O*sin_w*cos_i) )
    TIpar_B = a*( (sin_O*cos_w) + (cos_O*sin_w*cos_i) )
    TIpar_F = a*( (-cos_O*sin_w) - (sin_O*cos_w*cos_i) )
    TIpar_G = a*( (-sin_O*sin_w) + (cos_O*cos_w*cos_i) )
    
    return TIpar_A, TIpar_B, TIpar_F, TIpar_G


def compute_relative_orbit(tvals, angle_O_deg, angle_w_deg, angle_i_deg,
                                 a_axis, eccentricity, period, T0, q_mass):
    """
    Function to compute the relative astrometric displacements between the
    primary and the secondary star due to the orbit, for several time values,
    given the orbit's parameters, and the mass ratio between the two stars.
    
    The results I compute here are the displacements, in the DEC and RA
    directions, of the secondary's position with respect to the primary, in the
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
        q_mass: mass ratio of the two stars, m_secondary/m_primary

    OUTPUT:
        relorbit_displ_dec: relative astrometric displacement of the secondary
            star with respect to the primary in the declination direction
            [arcsec]
        relorbit_displ_ra: relative astrometric displacement of the secondary
            star with respect to the primary in the right ascention
            direction, in the appropriate rectilinear coordinates (i.e. taking
            into account the cos(delta) term using the nominal delta value)
            [arcsec]
    """
    
    tvals = np.atleast_1d(tvals)
    Nt = len(tvals)
    relorbit_displ_dec = np.empty(Nt)
    relorbit_displ_ra = np.empty(Nt)
    
    # Primary
    A, B, F, G = _get_ThieleInnes_from_params(a_axis, angle_O_deg, angle_w_deg,
                                              angle_i_deg)
    # Secondary
    A_sec, B_sec, F_sec, G_sec = \
        _get_ThieleInnes_from_params_secondary(a_axis, angle_O_deg, angle_w_deg,
                                              angle_i_deg, q_mass)
        
    for i,t in enumerate(tvals):
        X, Y = _get_XY_from_time(t, eccentricity, period, T0)
        relorbit_displ_dec[i] = A_sec*X + F_sec*Y - (A*X + F*Y)
        relorbit_displ_ra[i] = B_sec*X + G_sec*Y - (B*X + G*Y)
        
    return relorbit_displ_dec, relorbit_displ_ra


def compute_orbit_secondary(tvals, angle_O_deg, angle_w_deg_prim, angle_i_deg,
                            a_axis_prim, eccentricity, period, T0, q_mass):
    """
    Function to compute the astrometric displacements for the secondary due to
    the orbit, for several time values, given the *primary's orbit* parameters,
    and the mass ratio.
    
    The results I compute here are the displacements, in the DEC and RA
    directions, with respect to the centre of mass of the system, in the
    appropriate projected rectilinear coordinate system (i.e. that used in
    Wright&Howard).
    
    INPUT:
        tvals: times, either single value or array [years]
        angle_O_deg: position angle of the ascending node [degrees]
        angle_w_deg_prim: argument of periastron of the primary star [degrees]
        angle_i_deg: inclination of the orbit with respect to the sky [degrees]
        a_axis_prim: semi-major axis of the primary star's orbit [arcsec]
        eccentricity: eccentricity of orbit
        period: period of the orbit [years, or other consistent units]
        T0: time of periastron passage [years, or other consistent units]
        q_mass: mass ratio of the two stars, m_secondary/m_primary

    OUTPUT:
        orbit_displ_dec: astrometric displacement of the secondary star with
            respect to the CoM in the declination direction [arcsec]
        orbit_displ_ra: astrometric displacement of the secondary star with
            respect to the CoM in the right ascention direction, in the
            appropriate rectilinear coordinates (i.e. taking into account the
            cos(delta) term using the nominal delta value)  [arcsec]
    """
    
    tvals = np.atleast_1d(tvals)
    Nt = len(tvals)
    orbit_displ_dec = np.empty(Nt)
    orbit_displ_ra = np.empty(Nt)
    
    A, B, F, G = \
        _get_ThieleInnes_from_params_secondary(a_axis_prim, angle_O_deg,
                                               angle_w_deg_prim, angle_i_deg,
                                               q_mass)
    
    for i,t in enumerate(tvals):
        X, Y = _get_XY_from_time(t, eccentricity, period, T0)
        orbit_displ_dec[i] = A*X + F*Y
        orbit_displ_ra[i] = B*X + G*Y
        
    return orbit_displ_dec, orbit_displ_ra

# FUNCTIONS NEEDED TO DESCRIBE THE MOTION OF THE CENTRE OF MASS OF THE SYSTEM
    
def _compute_propmotion_lin(tvals, mu_alpha, mu_delta, t_ref):
    """
    Function to compute the term corresponding to the proper motion of the
    system assuming a simple linear model.
    
    INPUT:
        tvals: times, either single value or array [years]
        mu_alpha: linear proper motion in the rigth ascencion direction
            [arcsec/year]
        mu_delta: linear proper motion in the declination direction
            [arcsec/year]
        t_ref: time taken as reference (where proper motion is defined to be
            zero) [years]
        
    OUTPUT:
        pm_delta: astrometric displacements of the CoM of the system due to
            the proper motion in the declination direction [arcsec]
        pm_alpha: astrometric displacements of the CoM of the system due to
            the proper motion in the right ascention direction, in the
            appropriate rectilinear coordinates [arcsec]
        
    """
    pm_delta = mu_delta*(tvals - t_ref)
    pm_alpha = mu_alpha*(tvals - t_ref)

    return pm_delta, pm_alpha


def get_parallax_factors(tvals, skycoords_nominal, ephemeris='de430'):
    """
    Function to obtain the $\Pi_{\alpha}$, $\Pi_{\delta}$ parallax factors
    of the system at the given time values.
    
    These factors are computed following eqs. (53,54) of Wright&Howard, and
    represent the astrometric displacements due to parallax in each direction.
    
    INPUT:
        tvals: times, either single value or array [years]
        skycoords_nominal: nominal coordinates of the system
            [astropy.coordinates.SkyCoord object]
        ephemeris: name of the JPL ephemerides database to use to obtain the
            needed ICRS positions of the Earth. Default and recommended is
            'de430', which will need a download of ~115MB of data the first
            time this is executed in a system. For other options, see
            Astropy's help page:
            http://docs.astropy.org/en/stable/coordinates/solarsystem.html
            
    OUTPUT:
        Par_factor_dec: parallax factor (relative astrometric displacement)
            in the declination direction  [technically, AU]
        Par_factor_ra: parallax factor (relative astrometric displacement)
            in the right ascension direction [technically, AU]
    """
    
    # Obtain the cartesian representation of Earth's position in the
    # barycentric ICRS coordinates at the needed times
    earth_cartesian_repr = \
        coord.get_body_barycentric(body='earth',
                                   time=Time(tvals, format='decimalyear'),
                                   ephemeris=ephemeris)
        
    # Convert coordinates in units of AU to appropriate numpy array 
    earth_r = earth_cartesian_repr.xyz.to(u.AU).value
    
    sin_ra = np.sin(skycoords_nominal.ra)
    cos_ra = np.cos(skycoords_nominal.ra)
    sin_dec = np.sin(skycoords_nominal.dec)
    cos_dec = np.cos(skycoords_nominal.dec)
    
    Par_factor_ra = earth_r[0]*sin_ra - earth_r[1]*cos_ra
    Par_factor_dec = \
        (earth_r[0]*cos_ra + earth_r[1]*sin_ra)*sin_dec - earth_r[2]*cos_dec
    
    return Par_factor_dec, Par_factor_ra


def compute_com_motion(tvals, pi_parallax, mu_propmotion, rel_pos_ref,
                       parall_factors, t_ref):
    """
    Function to compute the positions of the centre of mass of the system at
    the given times.
    
    We compute the corresponding astrometric displacements in the appropriate
    projected rectilinear coordinate system used in Wright&Howard with respect
    to the nominal coordinates.
    
    The displacements computed contain three terms:
        * Proper motion term
        * Displacements due to the parallax
        * Change in the position of the CoM at the reference time with respect
            to the nominal coordinates
    
    INPUT:
        tvals: times, either single value or array [years]
        pi_parallax: parallax of the system [arcsec, technically arcsec/AU]
        mu_propmotion=(mu_delta, mu_alpha): linear proper motion of the system
            in each of the directions (DEC, RA) [arcsec]
        rel_pos_ref=(Ddec, Dra): reference position of the CoM of the system
            in the projected rectilinear coordinates, with respect to the
            nominal coordinates. The reference position is the position at
            the t_ref, if there were no parallax displacements [arcsec]
        parall_factors=(Pfactor_dec, Pfactor_ra): parallax factors
            corresponding to the times tvals, in each of the directions
            (DEC, RA) [technically, AU]
        t_ref: time taken as reference (where proper motion is defined to be
            zero) [years]
        
    OUTPUT:
        com_pos_delta: astrometric displacements of the CoM of the system
            in the declination direction [arcsec]
        com_pos_alpha: astrometric displacements of the CoM of the system
            in the right ascention direction, in the appropriate rectilinear
            coordinates [arcsec]
    """
    
    mu_delta, mu_alpha = mu_propmotion
    Ddec, Dra = rel_pos_ref
    Pfactor_dec, Pfactor_ra = parall_factors
    
    prop_motion_dec, prop_motion_ra = _compute_propmotion_lin(tvals, mu_alpha,
                                                              mu_delta, t_ref)
    
    parallax_dec = pi_parallax*Pfactor_dec
    parallax_ra = pi_parallax*Pfactor_ra
    
    com_pos_delta = prop_motion_dec + parallax_dec + Ddec
    com_pos_alpha = prop_motion_ra + parallax_ra + Dra
    
    return com_pos_delta, com_pos_alpha

    
# PUT EVERYTHING TOGETHER TO COMPUTE THE TOTAL ABSOLUTE MOTION OF EACH OF THE STARS
    
def compute_total_motion_primary(tvals, angle_O_deg, angle_w_deg, angle_i_deg,
                                 a_axis, eccentricity, period, T0, pi_parallax,
                                 mu_propmotion, rel_pos_ref, parall_factors,
                                 t_ref):
    """
    Function to compute the absolute position of the primary star at the
    given times, taking into account both the motion of the centre of mass,
    and the orbit.
    
    Basically, this is just a wrapper over compute_orbit_primary and 
    compute_com_motion.
    
    INPUT:
        tvals: times, either single value or array [years]
        angle_O_deg: position angle of the ascending node [degrees]
        angle_w_deg: argument of periastron of the primary star [degrees]
        angle_i_deg: inclination of the orbit with respect to the sky [degrees]
        a_axis: semi-major axis of the primary star's orbit [arcsec]
        eccentricity: eccentricity of orbit
        period: period of the orbit [years, or other consistent units]
        T0: time of periastron passage [years, or other consistent units]
        pi_parallax: parallax of the system [arcsec, technically arcsec/AU]
        mu_propmotion=(mu_delta, mu_alpha): linear proper motion of the system
            in each of the directions (DEC, RA) [arcsec]
        rel_pos_ref=(Ddec, Dra): reference position of the CoM of the system
            in the projected rectilinear coordinates, with respect to the
            nominal coordinates. The reference position is the position at
            the t_ref, if there were no parallax displacements [arcsec]
        parall_factors=(Pfactor_dec, Pfactor_ra): parallax factors
            corresponding to the times tvals, in each of the directions
            (DEC, RA) [technically, AU]
        t_ref: time taken as reference (where proper motion is defined to be
            zero) [years]
    
    OUTPUT:
        abs_pos_delta: absolute astrometric displacements of the primary star
            in the declination direction [arcsec]
        abs_pos_alpha: absolute astrometric displacements of the primary star
            in the right ascention direction, in the appropriate rectilinear
            coordinates [arcsec]
    """
    
    com_pos_delta, com_pos_alpha = \
        compute_com_motion(tvals, pi_parallax, mu_propmotion, rel_pos_ref,
                           parall_factors, t_ref)
        
    rel_orbit_pos_delta, rel_orbit_pos_alpha = \
        compute_orbit_primary(tvals, angle_O_deg, angle_w_deg, angle_i_deg,
                              a_axis, eccentricity, period, T0)
        
    abs_pos_delta = rel_orbit_pos_delta + com_pos_delta
    abs_pos_alpha = rel_orbit_pos_alpha + com_pos_alpha
    
    return abs_pos_delta, abs_pos_alpha


def compute_total_motion_secondary(tvals, angle_O_deg, angle_w_deg_prim,
                                   angle_i_deg, a_axis_prim, eccentricity,
                                   period, T0, q_mass, pi_parallax,
                                   mu_propmotion, rel_pos_ref, parall_factors,
                                   t_ref):
    """
    Function to compute the absolute position of the secondary star at the
    given times, taking into account both the motion of the centre of mass,
    and the orbit.
    
    Basically, this is just a wrapper over compute_orbit_secondary and 
    compute_com_motion.
    
    INPUT:
        tvals: times, either single value or array [years]
        angle_O_deg: position angle of the ascending node [degrees]
        angle_w_deg_prim: argument of periastron of the primary star [degrees]
        angle_i_deg: inclination of the orbit with respect to the sky [degrees]
        a_axis_prim: semi-major axis of the primary star's orbit [arcsec]
        eccentricity: eccentricity of orbit
        period: period of the orbit [years, or other consistent units]
        T0: time of periastron passage [years, or other consistent units]
        pi_parallax: parallax of the system [arcsec, technically arcsec/AU]
        q_mass: mass ratio of the two stars, m_secondary/m_primary
        mu_propmotion=(mu_delta, mu_alpha): linear proper motion of the system
            in each of the directions (DEC, RA) [arcsec]
        rel_pos_ref=(Ddec, Dra): reference position of the CoM of the system
            in the projected rectilinear coordinates, with respect to the
            nominal coordinates. The reference position is the position at
            the t_ref, if there were no parallax displacements [arcsec]
        parall_factors=(Pfactor_dec, Pfactor_ra): parallax factors
            corresponding to the times tvals, in each of the directions
            (DEC, RA) [technically, AU]
        t_ref: time taken as reference (where proper motion is defined to be
            zero) [years]
    
    OUTPUT:
        abs_pos_delta: absolute astrometric displacements of the secondary star
            in the declination direction [arcsec]
        abs_pos_alpha: absolute astrometric displacements of the secondary star
            in the right ascention direction, in the appropriate rectilinear
            coordinates [arcsec]
    """
    
    com_pos_delta, com_pos_alpha = \
        compute_com_motion(tvals, pi_parallax, mu_propmotion, rel_pos_ref,
                           parall_factors, t_ref)
        
    rel_orbit_pos_delta, rel_orbit_pos_alpha = \
        compute_orbit_secondary(tvals, angle_O_deg, angle_w_deg_prim,
                                angle_i_deg, a_axis_prim, eccentricity,
                                period, T0, q_mass)
        
    abs_pos_delta = rel_orbit_pos_delta + com_pos_delta
    abs_pos_alpha = rel_orbit_pos_alpha + com_pos_alpha
    
    return abs_pos_delta, abs_pos_alpha
    
    
    
    
    
    