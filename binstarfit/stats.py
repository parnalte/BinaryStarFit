#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:37:06 2017

@author: pablo

Contains the statistical modelling (definition of priors, likelihoods,
sampling, ...)
"""

import numpy as np
import scipy.stats as st

from binstarfit import model


# List containing the names (keys) of all the parameters to be used.
# This is here in part for reference. The parameter arrays that will be passed
# around will always follow this same ordering
param_list_all = ['Omega_angle',
                  'omega_angle',
                  'i_angle',
                  'a_axis',
                  'ecc',
                  'period',
                  'T_0',
                  'mu_delta',
                  'mu_alpha',
                  'pi_p',
                  'Ddelta_ref',
                  'Dalpha_ref',
                  'q_m']


# Default prior parameters.
# For now, this only works for some of the parameters, while for the other
# parameters, we fix the prior in the code.
# In all cases the distribution type is fixed, but in the cases listed here,
# we allow for changes in the distribution's parameters
# TODO: add some more flexibility here
# In this dict, the loc/scale parameters correspond to the respective ones
# in the distribution functions in scipy.stats
prior_params_default = {}

# For primary axis in arcsec (a): HalfNormal distribution
prior_params_default['a_axis'] = {'loc': 0, 'scale': 0.1}

# For period in years (P): Uniform distribution
prior_params_default['period'] = {'loc': 0, 'scale': 25}

# For DEC proper motion in arcsec/year (mu_alpha): Normal distribution
prior_params_default['mu_delta'] = {'loc': 0, 'scale': 0.3}

# For RA proper motion in arcsec/year (mu_alpha): Normal distribution
prior_params_default['mu_alpha'] = {'loc': 0, 'scale': 0.3}

# For parallax in arcsec (pi_p): HalfNormal distribution
prior_params_default['pi_p']= {'loc': 0, 'scale': 0.1}

# For DEC of reference position in arcsec (Ddelta_ref): Normal distribution
prior_params_default['Ddelta_ref'] = {'loc': 0, 'scale': 0.5}

# For RA of reference position in arcsec (Ddelta_ref): Normal distribution
prior_params_default['Dalpha_ref'] = {'loc': 0, 'scale': 0.5}

# For ratio of masses (q_m): HalfNormal distribution
prior_params_default['q_m'] = {'loc': 0, 'scale': 1}




def lnprior(theta, ref_time, fit_qm=False, prior_params=prior_params_default):
    """
    Function to compute the value of ln(prior) for a given set of parameters.
    
    We compute the prior using fixed definitions for the prior distributions
    of the parameters, allowing some optional parameters for some of them via
    the 'prior_params' dictionary. Except for T0 and P the priors on each of
    the parameters is taken to be independent of each other, and defined in the
    following way:
        
        - Omega_angle: uniform between 0-180 degrees
        - omega_angle: uniform between -180 - 180 degrees
        - i_angle: uniform between 0 - 90 degrees
        - a_axis: half-normal with optional loc/scale
        - ecc: uniform between 0 - 1
        - period: uniform with optional loc/scale
        - T0: uniform between ref_time and period (this is to restrict the
            result to a single-valued parameter)
        - mu_delta, mu_alpha: normal with optional loc/scale
        - pi_p: halfnormal with optional loc/scale
        - Ddelta_ref, Dalpha_ref: normal with optional loc/scale
        - q_m (only considered if fit_qm=True): halfnormal with optional
            loc/scale
    
    INPUT:
        theta: array of parameters (ndim=12 or 13), contains the model
            parameters following the ordering defined by param_list_all 
            (with or without q_m at the end)
        ref_time: reference time, used to define the prior on T0 [years]
        fit_qm: whether we are fitting for the mass ratio q_m or not
        prior_params: dictionary containing the optional parameters for some
            of the model parameters (should be defined as prior_params_default)
            
    OUTPUT:
        lprior: value of ln(prior) at this position in parameter space
    
    """
    
    if fit_qm:
        Omega_angle, omega_angle, i_angle, a_axis, ecc, period, T0, mu_delta, \
            mu_alpha, pi_p, Ddelta_ref, Dalpha_ref, q_m = theta
    else:
         Omega_angle, omega_angle, i_angle, a_axis, ecc, period, T0, mu_delta, \
            mu_alpha, pi_p, Ddelta_ref, Dalpha_ref = theta
        
    lprior = 0
    lprior += st.uniform.logpdf(Omega_angle, loc=0, scale=180)
    lprior += st.uniform.logpdf(omega_angle, loc=-180, scale=360)
    lprior += st.uniform.logpdf(i_angle, loc=0, scale=90)
    
    lprior += st.halfnorm.logpdf(a_axis, **prior_params['a_axis'])
    
    lprior += st.uniform.logpdf(ecc, loc=0, scale=1)
    
    lprior += st.uniform.logpdf(period, **prior_params['period'])
    lprior += st.uniform.logpdf(T0, loc=ref_time, scale=period)
    
    lprior += st.norm.logpdf(mu_delta, **prior_params['mu_delta'])
    lprior += st.norm.logpdf(mu_alpha, **prior_params['mu_alpha'])

    lprior += st.halfnorm.logpdf(pi_p, **prior_params['pi_p'])
    
    lprior += st.norm.logpdf(Ddelta_ref, **prior_params['Ddelta_ref'])
    lprior += st.norm.logpdf(Dalpha_ref, **prior_params['Dalpha_ref'])
    
    if fit_qm:
        lprior += st.halfnorm.logpdf(q_m, **prior_params['q_m'])
    
    return lprior


def lnlikelihood(theta, ref_time, data_abs_primary,
                 data_abs_secondary=None, data_relative=None):
    """
    Function to compute the value of the ln(likelihood) of the data for a
    given set of parameters. In all cases, we assume Normal, indpendent errors
    on the RA and DEC measurements.
    
    For now, we require to have some data on the absolute positions of the
    primary, and we decide whether q_m is one of the parameters to fit
    depending on whether we have some data for the relative orbit and/or
    the absolute positions of the secondary.
    
    INPUT:
        theta: array of parameters (ndim=12 or 13), contains the model
            parameters following the ordering defined by param_list_all 
            (with or without q_m at the end)
        ref_time: reference time [years]
        data_abs_primary: dictionary containing the data for the absolute
            positions of the primary star. Contains the following keys:
            - time: times of observations
            - dec: astrometric displacements in the DEC direction, in the 
                appropriate projected rectilinear coordinate system [arcsec]
            - ra: astrometric displacements in the RA direction, in the 
                appropriate projected rectilinear coordinate system [arcsec]
            - dec_err: error (1-sigma) of the measurements in the DEC direction
                [arcsec]
            - ra_err: error (1-sigma) of the measurements in the RA direction
                [arcsec]
            - Pfactor_dec: parallax factors corresponding to time in the DEC
                direction
            - Pfactor_ra: parallax factors corresponding to time in the RA
                direction
        data_abs_secondary (optional): dictionary containing the data for
            the absolute positions of the secondary star. Same keys as
            data_abs_primary
        data_relative (optional): dictionary containing the data for the
            relative positions (secondary with respect to primary). Same keys
            as data_abs_primary, except for the parallax factors which are not
            needed
    
    OUTPUT:
        llike: value of the ln(likelihood) of the data at this position in
            parameter space
    """
    
    if (data_abs_secondary is not None) or (data_relative is not None):
        Omega_angle, omega_angle, i_angle, a_axis, ecc, period, T0, mu_delta, \
            mu_alpha, pi_p, Ddelta_ref, Dalpha_ref, q_m = theta
    else:
         Omega_angle, omega_angle, i_angle, a_axis, ecc, period, T0, mu_delta, \
            mu_alpha, pi_p, Ddelta_ref, Dalpha_ref = theta
    
    # Get contribution to chi2 from the measurements of the primary's absolute
    # position
    model_abs_delta, model_abs_alpha = \
        model.compute_total_motion_primary(tvals=data_abs_primary['time'],
            angle_O_deg=Omega_angle, angle_w_deg=omega_angle,
            angle_i_deg=i_angle, a_axis=a_axis, eccentricity=ecc,
            period=period, T0=T0, pi_parallax=pi_p,
            mu_propmotion=(mu_delta, mu_alpha),
            rel_pos_ref=(Ddelta_ref, Dalpha_ref),
            parall_factors=(data_abs_primary['Pfactor_dec'],
                            data_abs_primary['Pfactor_ra']),
            t_ref=ref_time)
            
    chi2_abs_delta = (( (model_abs_delta - data_abs_primary['dec'])/data_abs_primary['dec_err'] )**2).sum()
    chi2_abs_alpha = (( (model_abs_alpha - data_abs_primary['ra'])/data_abs_primary['ra_err'] )**2).sum()
    chi2_abs = chi2_abs_delta + chi2_abs_alpha
    
    # Get contribution from secondary's absolute position
    if data_abs_secondary is None:
        chi2_sec = 0
        
    else:
        model_sec_delta, model_sec_alpha = \
            model.compute_total_motion_secondary(tvals=data_abs_secondary['time'],
                angle_O_deg=Omega_angle, angle_w_deg_prim=omega_angle,
                angle_i_deg=i_angle, a_axis_prim=a_axis, eccentricity=ecc,
                period=period, T0=T0, q_mass=q_m, pi_parallax=pi_p,
                mu_propmotion=(mu_delta, mu_alpha),
                rel_pos_ref=(Ddelta_ref, Dalpha_ref),
                parall_factors=(data_abs_secondary['Pfactor_dec'],
                                data_abs_secondary['Pfactor_ra']),
                t_ref=ref_time)
                
        chi2_sec_delta = (( (model_sec_delta - data_abs_secondary['dec'])/data_abs_secondary['dec_err'] )**2).sum()
        chi2_sec_alpha = (( (model_sec_alpha - data_abs_secondary['ra'])/data_abs_secondary['ra_err'] )**2).sum()
        chi2_sec = chi2_sec_delta + chi2_sec_alpha
        
    # Get contribution from relative orbit measurements
    if data_relative is None:
        chi2_rel = 0
    else:
        model_rel_delta, model_rel_alpha = \
            model.compute_relative_orbit(tvals=data_relative['time'],
                angle_O_deg=Omega_angle, angle_w_deg=omega_angle,
                angle_i_deg=i_angle, a_axis=a_axis, eccentricity=ecc,
                period=period, T0=T0, q_mass=q_m)
            
        chi2_rel_delta = (( (model_rel_delta - data_relative['dec'])/data_relative['dec_err'] )**2).sum()
        chi2_rel_alpha = (( (model_rel_alpha - data_relative['ra'])/data_relative['ra_err'] )**2).sum()
        chi2_rel = chi2_rel_delta + chi2_rel_alpha
        
    return -0.5*(chi2_abs + chi2_sec + chi2_rel)
