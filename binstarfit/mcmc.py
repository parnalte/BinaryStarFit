#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 11:42:22 2017

@author: pablo

Contains the functions to run the MCMC sampling

"""
import os

import numpy as np

import emcee

from binstarfit import stats, io


def _iterate_sampler(sampler, nsteps, p0, chainfile=None):
    """
    Wrapper function to iterate the PTSampler for nsteps steps, while printing
    out a progress report and, if chainfile is given, write out the T=1 chain
    to file.
    (chainfile is always appended, not overwritten, and no header is added).
    
    This wrapper function is basically taken from emcee's docs.
    
    This functions returns the position (for all walkers and temps) obtained
    from the last iteration, so that it can be reused for continuing the
    sampling.
    """
    
    # First check things are as they should
    assert p0.shape == (sampler.ntemps, sampler.nwalkers, sampler.dim)
    
    # Run the sampling
    for i, (pos, lnprob, lnlike) in enumerate(sampler.sample(p0, iterations=nsteps)):
        if (i+1) % 100 == 0:
            print("{0:5.1%}".format(float(i+1) / nsteps), end=' ')
        
        if chainfile is not None:
            pos_t1 = pos[0]
            lnprob_t1 = lnprob[0]

            with open(chainfile, "a") as f:
                for k in range(pos_t1.shape[0]):
                    f.write("%d  %s  %g\n" % (k, np.array_str(pos_t1[k], max_line_width=1000)[1:-1],
                                              lnprob_t1[k]))
                
    return pos


def main(ntemps, nwalkers, nburnin, niter, outchain_file, outpos_file,
         ref_time, skycoords_nominal, data_fname_prim,
         data_fname_sec=None, data_fname_rel=None, rel_is_primary=False,
         prior_params=stats.prior_params_default, rseed=None, threads=1,
         pstart_file=None, ephemeris='de430'):
    """
    Main function to perform the MCMC fit to the data using emcee's PTSampler.
    
    This will read in the available data, initialize the sampler, run it for
    a number of burn-in iterations (not saved), and then run it to obtain a
    sample of 'good' samples to map the posterior distribution.
    
    INPUT:
        - ntemps: the number of temperatures to be used by the PT-sampler
        - nwalkers: the number of walkers to be used by the PT-sampler at each
            temperature
        - nburnin: the number of initial burn-in iterations of the sampler to
            be discarded
        - niter: the number of iterations of the sampler to be kept *after*
            the burn-in phase
        - outchain_file: the name of the file where we will store the output
            chain (at T=1, corresponding to the actual posterior). The file
            should not exist, and will be created.
        - outpos_file: name of the file where we will save (in numpy's format)
            the last position of the walkers (at all temperatures), so that
            the sampling can be continued from it if more samples are needed
        - ref_time: reference time for the definition of our model parameters
            [years]
        - skycoords_nominal: nominal coordinates of the system analysed, needed
            for the definition of coordinates in our model 
            [astropy's SkyCoord object]
        - data_fname_prim: name of the file containing the observational data
            for the absolute positions of the primary star
        - data_fname_sec (optional): name of the file containing the
            observational data for the absolute positions of the secondary star
        - data_fname_rel (optional): name of the file containing the
            observational data for the relative orbit
        - rel_is_primary: whether the relative data refers to the orbit of the
            primary wrt the secondary (default: False, relative data refers
            to the orbit of the secondary wrt the primary)
        - prior_params: dictionary containing the definition of the prior for
            some of the model parameters (see definitions in stats.py)
        - rseed (optional but recommended): seed to initialise the random
            number generator
        - threads (optional): number of threads to use in the computation
        - pstart_file (optional): if given, it should contain a starting
            position for the walkers in the sampler, saved in numpy's format.
            This can be used to continue a previous (partial) run, e.g. if
            we found out it wasn't fully converged. If not given, the initial
            positions of the walkers will be generated following the prior.
        - ephemeris: name of the JPL ephemerides database to use to obtain the
            needed ICRS positions of the Earth. Default and recommended is
            'de430'.

            
    """
    
    if os.path.exists(outchain_file):
        raise ValueError("File %s already exists, I will not overwrite it!!" 
                         % outchain_file)
        
    if os.path.exists(outpos_file):
        raise ValueError("File %s already exists, I will not overwrite it!!"
                         % outpos_file)
        
    # TODO: add some assert's here to make sure input parameters make sense!!
    assert ntemps > 0
    assert nwalkers > 12
    assert nburnin >= 0
    assert niter > 0
    
    if rseed is not None:
        np.random.seed(rseed)
        
    # Read in data 
    
    # Primary's absolute positions
    Nobs_prim, datadict_prim = \
        io.read_absolute_data(fname=data_fname_prim,
                              skycoords_nominal=skycoords_nominal,
                              ephemeris=ephemeris)
    print("Correctly read %d observations for the primary from file %s"
          % (Nobs_prim, data_fname_prim))
    
    # If available, secondary's absolute positions
    fit_qm = False
    Nobs_sec = 0
    datadict_sec = None
    if data_fname_sec is not None:
        Nobs_sec, datadict_sec = \
            io.read_absolute_data(fname=data_fname_sec,
                                  skycoords_nominal=skycoords_nominal,
                                  ephemeris=ephemeris)
        print("Correctly read %d observations for the secondary from file %s"
              % (Nobs_sec, data_fname_sec))
        fit_qm = True
        
    # If available, relative positions
    Nobs_rel = 0
    datadict_rel = None
    if data_fname_rel is not None:
        Nobs_rel, datadict_rel = \
            io.read_relative_data(fname=data_fname_rel,
                                  primary_wrt_secondary=rel_is_primary)
        print("Correctly read %d observations for the relative orbit from file %s"
              % (Nobs_rel, data_fname_rel))
        fit_qm = True
        
    # Show basic properties of the fit
    Nobs_total = Nobs_prim + Nobs_sec + Nobs_rel
    Ndata = 2*Nobs_total
    
    if fit_qm:
        Ndim = 13
    else:
        Ndim = 12
        
    print("We have a total of %d datapoints (%d observations) for a model with"
          " %d parameters." % (Ndata, Nobs_total, Ndim))
    print("Therefore, the number of degrees of freedom is %d" % (Ndata-Ndim))
    
    
    # Define the starting positions for the walkers
    if pstart_file is not None:
        p0 = np.load(pstart_file)
        assert p0.shape == (ntemps, nwalkers, Ndim)
        
    else:
        p0 = stats.generate_initial_positions(ntemps, nwalkers, ref_time,
                                              fit_qm, prior_params)
        
    # Define the sampler object
    sampler = emcee.PTSampler(ntemps=ntemps, nwalkers=nwalkers, dim=Ndim,
                              logl=stats.lnlikelihood, logp=stats.lnprior,
                              loglargs=(ref_time, datadict_prim, datadict_sec,
                                        datadict_rel),
                              logpargs=(ref_time, fit_qm, prior_params),
                              threads=threads)
                              
    # Now, run the burn-in iterations (this should be the longest part)
    print("Now, we will run the PTSampler for %d burn-in iterations that will"
          " be discarded." % nburnin)
    
    # Note, we do not pass a chainfile here, as we do not want to save this part
    p_burnt = _iterate_sampler(sampler=sampler, nsteps=nburnin, p0=p0)
    
    # And, once the chains have 'burned in', do the final run
    print("Now, we will run the PTSampler for %d iterations to get the final chain" % niter)
    
    # Add a header to the chain file
    if fit_qm:
        with open(outchain_file, 'w') as f:
            f.write("walker Omega omega i a e P T0 mu_delta mu_alpha pi_par Ddec Dracd q_mass log_posterior\n")
    else:
        with open(outchain_file, 'w') as f:
            f.write("walker Omega omega i a e P T0 mu_delta mu_alpha pi_par Ddec Dracd log_posterior\n")
    
    # Note, we want to save this result in the outchain_file. 
    # Also, now the starting position is that coming from the previous step
    pfinal = _iterate_sampler(sampler=sampler, nsteps=niter, p0=p_burnt,
                              chainfile=outchain_file)
    
    # Save final state to file
    np.save(outpos_file, pfinal)
    
    # Work is finished, output some info and basic assessment of chain
    print("This run of the MCMC sampling has finished.")
    print("The complete chain (for T=1) has been saved to file %s" % outchain_file)
    print("The final position of the sampler (for all temps) has been saved to file %s" % outpos_file)
    print()
    print("Mean acceptance fraction for the different temperatures:")
    print(sampler.acceptance_fraction.mean(axis=1))
    print()
    print("Temp-swap acceptance fraction for the different temperatures:")
    print(sampler.tswap_acceptance_fraction)
    print("****************************************************************")
    
    
    





