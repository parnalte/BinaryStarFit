#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:14:42 2017

@author: pablo

Contains functions to deal with reading in data files
"""

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.coordinates as coord

from binstarfit import model




def read_absolute_data(fname, skycoords_nominal, ephemeris='de430'):
    """
    Function to read in a data file containing absolute astrometric
    observations for one star, and output a dictionary containing the needed
    keys/values for the stats.lnlikelihood function.
    This includes converting the astrometric displacements to the appropriate
    projected rectiliniar coordinates of Wright&Howard, and to arcseconds,
    and also computing the needed parallax factors at the observations' times.
    
    INPUT:
        - fname: name of the .csv file containing the observational data
            (see NOTES below for the format)
        - skycoords_nominal: nominal coordinates of the system, needed for the
            definition of the projected coordinates
            [astropy.coordinates.SkyCoord object]
        - ephemeris: name of the JPL ephemerides database to use to obtain the
            needed ICRS positions of the Earth. Default and recommended is
            'de430'.
            
    OUTPUT:
        - Nd: number of observations read
        - data_dict: dictionary containing the keys 'time', 'dec', 'ra',
            'dec_err', 'ra_err', 'Pfactor_dec', 'Pfactor_ra' as required by
            function stats.lnlikelihood
    
    NOTES:
       The input data file should be formatted as a comma-separated file,
       with the following columns (should include the header):
           - Epoch: time of the observation in decimalyear format
           - RA: right ascension of the star in HMS format
           - RA_err: error on the RA of the star, in *seconds* (i.e., following
               same format as RA)
           - DEC: declination of the star in DMS format
           - DEC_err: error on the DEC of the star, in *arcseconds* (i.e.,
               following the same format as DEC)
       There is an example file 'ABDorA_pos.csv" in the data folder.
       TODO: allow for more flexible formatting of the input data.
    """
    
    dfdata = pd.read_csv(fname)
    Nd = len(dfdata)
    
    data_dict = {}
    
    # Read in time column
    data_dict['time'] = dfdata['Epoch'].values
    
    # Read in and convert RA/DEC columns
    observed_coords = coord.SkyCoord(dfdata["RA"], dfdata["DEC"],
                                     unit=(u.hourangle, u.deg))

    rel_dec_obs = observed_coords.dec - skycoords_nominal.dec
    data_dict['dec'] = rel_dec_obs.to(u.arcsec).value
    
    rel_ra_obs = \
        (observed_coords.ra - skycoords_nominal.ra)*np.cos(skycoords_nominal.dec) 
    data_dict['ra'] = rel_ra_obs.to(u.arcsec).value

    # Read in and convert error columns
    data_dict['dec_err'] = dfdata['DEC_err'].values
    
    err_ra_seconds = dfdata['RA_err'].values
    data_dict['ra_err'] = \
        15.*err_ra_seconds*np.cos(skycoords_nominal.dec).value
        
    # Get parallax factors
    data_dict['Pfactor_dec'], data_dict['Pfactor_ra'] = \
        model.get_parallax_factors(data_dict['time'], skycoords_nominal,
                                   ephemeris)
        
    return Nd, data_dict

    
def read_relative_data(fname, primary_wrt_secondary=False):
    """
    Function to read in a data file containing relative astrometric
    observations for a binary system, and output a dictionary containing the
    needed keys/values for the stats.lnlikelihood function.
    We assume the data in the input file is already  in arcsec, so no 
    conversion is required.
    
    INPUT:
        - fname: name of the .csv file containing the observational data
            (see NOTES below for the format)
        - primary_wrt_secondary: whether the file lists the difference in
            coordinates of the primary with respect to the secondary
            (the default is the opposite: listing the position of the secondary
            with respect to the primary)
            
    OUTPUT:
        - Nd: number of observations read
        - data_dict: dictionary containing the keys 'time', 'dec', 'ra',
            'dec_err', 'ra_err' as required by function stats.lnlikelihood
    
    NOTES:
       The input data file should be formatted as a comma-separated file,
       with the following columns (should include the header):
           - Epoch: time of the observation in decimalyear format
           - D_RA: difference in the right ascension of the stars in arcsec
           - D_RA_err: error on D_RA, in arcseconds
           - D_DEC: difference in the declination of the stars in arcsec
           - D_DEC_err: error on D_DEC, in arcseconds
       There is an example file 'ABDorA-C_relpos.csv" in the data folder (in
       this particular case, this file should be read using
       primary_wrt_secondary=True)
       TODO: allow for more flexible formatting of the input data.
    """
    
    dfdata = pd.read_csv(fname)
    Nd = len(dfdata)
    
    data_dict = {}
    
    # Read in columns
    data_dict['time'] = dfdata['Epoch'].values
    data_dict['dec'] = dfdata['D_DEC'].values
    data_dict['ra'] = dfdata['D_RA'].values
    data_dict['dec_err'] = dfdata['D_DEC_err'].values
    data_dict['ra_err'] = dfdata['D_RA_err'].values
    
    if primary_wrt_secondary:
        data_dict['dec'] = -data_dict['dec']
        data_dict['ra'] = -data_dict['ra']
        
    return Nd, data_dict

    