#!/usr/bin/env python

################################################################################
# created by Luka Lambrecht and Marek Niedziela                                #
# a set of functions to process dataframes used for ML4DQM working group       #
# 8th May, 2020                                                                #
################################################################################

import numpy as np
import pandas as pd
import random as rn
import os
import json
from ast import literal_eval
from plotters import *

def resample_similar_lico(rhist,doplot=True,nresamples=0,nonnegative=False,keeppercentage=1.,whitenoisefactor=0.):
    # take linear combinations of similar histograms

    # advantages: no assumptions on noise
    # disadvantages: sensitive to outlying histograms (more than with averaging)
    
    # set some parameters
    if nresamples==0: nresamples=int(len(rhist)/10)
    (nhists,nbins) = rhist.shape
    
    # get array of moments (used to define similar histograms)
    binwidth = 1./nbins
    bincenters = np.linspace(binwidth/2,1-binwidth/2,num=nbins,endpoint=True)
    orders = [0,1,2]
    moments = np.zeros((nhists,len(orders)))
    for i,j in enumerate(orders): moments[:,i] = moment(bincenters,rhist,j)
    
    # make resampled histograms
    ghist = np.zeros((nresamples,nbins))
    randint = np.random.choice(np.arange(0,len(rhist)),size=nresamples,replace=True)
    for i,j in enumerate(randint):
        # select similar histograms
        thisdiff = moments_correlation_vector(moments,j)
        #thisdiff = mse_correlation_vector(rhist,j)
        threshold = np.percentile(thisdiff,keeppercentage)
        simindices = np.nonzero(np.where(thisdiff<threshold,1,0))[0]
        ghist[i,:] = random_lico(rhist[simindices,:])
        ghist[i,:] += whitenoise(nbins,ghist[i,:]*whitenoisefactor)
        if nonnegative: ghist[i,:] = np.maximum(0,ghist[i,:])
    nsim = len(simindices)
    print('Note: linear combination is taken between '+str(nsim)+' histograms.')
    print('If this number is too low, histograms might be too similar for combination to have effect.')
    print('If this number is too high, systematic shifts of histogram shapes are included into the combination')
        
    # plot examples of good and bad histograms
    # use only those histograms from real data that were used to create the resamples
    if doplot: plot_data_good_bad(50,rhist[randint],ghist,np.array([[0]]))
        
    return ghist


def resample_similar_fourier_noise(rhist,doplot=True,nresamples=0,nonnegative=False,keeppercentage=1.,whitenoisefactor=0.):
    # apply fourier noise on mean histogram, 
    # where the mean is determined from a set of similar-looking histograms
    # input args:
    # - rhist: numpy array of shape (nhists,nbins)
    # - doplot: boolean whether or not to do some plotting
    # - nresamples: number of samples to be drawn
    # - nonnegative: boolean whether or not to put all bins to minimum zero after applying noise
    # - keeppercentage: percentage (between 1 and 100) of histograms in rhist that are 'similar' to a given histogram
    # - whitenoisefactor: factor (between 0 and 1) of white noise amplitude (relative to histogram bin content)

    # advantages: most of fourier_noise_on_mean but can additionally handle shifting histograms,
    #             apart from fourier noise, also white noise can be applied.
    # disadvantages: does not filter out odd histograms as long as enough other odd histograms look more or less similar
    
    # set some parameters
    if nresamples==0: nresamples=int(len(rhist)/10)
    (nhists,nbins) = rhist.shape
    
    # get array of moments (used to define similar histograms)
    binwidth = 1./nbins
    bincenters = np.linspace(binwidth/2,1-binwidth/2,num=nbins,endpoint=True)
    orders = [0,1,2]
    moments = np.zeros((nhists,len(orders)))
    for i,j in enumerate(orders): moments[:,i] = moment(bincenters,rhist,j)
    
    # make resamples
    ghist = np.zeros((nresamples,nbins))
    randint = np.random.choice(np.arange(0,len(rhist)),size=nresamples,replace=True)
    for i,j in enumerate(randint):
        # select similar histograms
        thisdiff = moments_correlation_vector(moments,j)
        #thisdiff = mse_correlation_vector(rhist,j)
        threshold = np.percentile(thisdiff,keeppercentage)
        simindices = np.nonzero(np.where(thisdiff<threshold,1,0))[0]
        ghist[i,:] = fourier_noise_on_mean(rhist[simindices,:],doplot=False,
                                         ngood=1,nbad=1,nonnegative=nonnegative)[0][0,:]
        ghist[i,:] += whitenoise(nbins,ghist[i,:]*whitenoisefactor)
        if nonnegative: ghist[i,:] = np.maximum(0,ghist[i,:])
    nsim = len(simindices)
    print('Note: mean and std calculation is performed on '+str(nsim)+' histograms.')
    print('If this number is too low, histograms might be too similar for averaging to have effect.')
    print('If this number is too high, systematic shifts of histogram shapes are included into the averaging.')
        
    # plot examples of good and bad histograms
    # use only those histograms from real data that were used to create the resamples
    if doplot: plot_data_good_bad(50,rhist[randint],ghist,np.array([[0]]))
        
    return ghist

def resample_bin_per_bin(rhist,doplot=True,nresamples=0,nonnegative=False,smoothinghalfwidth=0):
    # do resampling from bin-per-bin probability distributions

    # advantages: no arbitrary noise modeling
    # disadvantages: bins are considered independent, shape of historams not taken into account,
    #                does not work well on small number of input histograms, 
    #                does not work well on histograms with systematic shifts
    # mostly deprecated, do not use this method.
    
    if nresamples==0: nresamples=int(len(rhist)/10)
    nbins = rhist.shape[1]
    
    # generate data
    ghist = np.zeros((nresamples,nbins))
    for i in range(nbins):
        col = np.random.choice(rhist[:,i],size=nresamples,replace=True)
        ghist[:,i] = col
        
    # apply smoothing to compensate partially for bin independence
    if smoothinghalfwidth>0: ghist = smoother(ghist,halfwidth=smoothinghalfwidth)

    # plot examples of good and bad histograms
    if doplot: plot_data_good_bad(50,rhist,ghist,np.array([[0]]))
    
    return ghist



def fourier_noise_on_mean(rhist,doplot=True,ngood=0,nbad=0,nonnegative=False):
    # apply fourier noise on the bin-per-bin mean histogram,
    # with amplitude scaling based on bin-per-bin std histogram.
    # input args:
    # - rhist: numpy array of shape (nhists,nbins)
    # - doplot: boolean whether to do some plotting
    # - ngood and nbad: integers, number of samples to draw with 'good' and 'bad' noise
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise

    # advantages: mean histogram is almost certainly 'good' because of averaging, eliminate bad histograms
    # disadvantages: deviations from mean are small, does not model systematic shifts by lumi.
    
    if ngood==0: ngood=int(len(rhist)/10)
    if nbad==0: nbad=int(len(rhist)/10)

    # get mean and std histogram
    histmean = np.mean(rhist,axis=0)
    histstd = np.std(rhist,axis=0)
    nbins = len(histmean)

    # plot examples of histograms mean, and std
    if doplot:
        nplot = min(200,len(rhist))
        flatindex = np.linspace(0,len(rhist),num=len(rhist),endpoint=False)
        randint = np.random.choice(flatindex,size=nplot,replace=False)
        plt.figure()
        for i in randint: plt.plot(rhist[int(i),:],color='b',alpha=1)
        plt.plot(histmean,color='black',label='mean')
        plt.plot(histmean-histstd,color='r',label='pm 1 std')
        plt.plot(histmean+histstd,color='r')
        plt.legend()
    
    # generate data
    ghist = np.zeros((ngood,nbins))
    bhist = np.zeros((nbad,nbins))
    for i in range(ngood): ghist[i,:] = histmean+goodnoise(nbins,histstd)
    for i in range(nbad): bhist[i,:] = histmean+badnoise(nbins,histstd)
    if nonnegative:
        ghist = np.where(ghist>0,ghist,0)
        bhist = np.where(bhist>0,bhist,0)
        
    # plot examples of good and bad histograms
    if doplot: 
        noise_examples = []
        for i in range(5): noise_examples.append(goodnoise(nbins,histstd))
        plot_noise(np.array(noise_examples),histstd)
        plot_data_good_bad(50,rhist,ghist,bhist)
    
    return (ghist,bhist) 




def fourier_noise_on_random(rhist,doplot=True,ngood=0,nbad=0,nonnegative=False):
    # apply fourier noise on randomly chosen histograms,
    # with simple flat amplitude scaling.
    # input args: similar to fourier_noise_on_mean

    # advantages: resampled histograms will have statistically same features as original
    # disadvantages: also 'bad' histograms will be resampled
    
    if ngood==0: ngood=int(len(rhist)/10)
    if nbad==0: nbad=int(len(rhist)/10)
    nbins = rhist.shape[1]
    # use histogram/stdfactor as std histogram...
    # the value is obtained by comparing distributions of real and resampled stable histograms.
    # so far cannot use sqrt(histogram) (i.e. poisson error) since normalized histograms are used.
    stdfactor = 15. 
    
    # generate data
    ghist = np.zeros((ngood,nbins))
    bhist = np.zeros((nbad,nbins))
    randint_good = np.random.choice(np.arange(0,len(rhist)),size=ngood,replace=True)
    for i,j in enumerate(randint_good): ghist[i,:] = rhist[j,:]+goodnoise(nbins,rhist[j,:]/stdfactor)
    randint_bad = np.random.choice(np.arange(0,len(rhist)),size=nbad,replace=True)
    for i,j in enumerate(randint_bad): bhist[i,:] = rhist[j,:]+badnoise(nbins,rhist[j,:]/stdfactor)
    if nonnegative:
        ghist = np.where(ghist>0,ghist,0)
        bhist = np.where(bhist>0,bhist,0)
        
    # plot examples of good and bad histograms
    if doplot: 
        noise_examples = []
        for i in range(5): noise_examples.append(goodnoise(nbins,rhist[randint_good[-1],:]/10.))
        plot_noise(np.array(noise_examples),rhist[randint_good[-1],:]/10.)
        plot_data_good_bad(50,rhist[randint_good,:],ghist,bhist)
    
    return (ghist,bhist)


# functions for calculating moments of a histogram

def moment(bins,counts,order):
    ### get n-th central moment of a histogram
    # - bins is a 1D or 2D np array holding the bin centers
    # - array is a 2D np array containing the bin counts
    if len(bins.shape)==1:
        bins = np.tile(bins,(len(counts),1))
    if not bins.shape == counts.shape:
        print('### ERROR ###: bins and counts do not have the same shape!')
        return None
    if len(bins.shape)==1:
        bins = np.array([bins])
        counts = np.array([counts])
    if order==0: # return maximum
        return np.nan_to_num(np.max(counts,axis=1))
    return np.nan_to_num(np.divide(np.sum(np.multiply(counts,np.power(bins,order)),axis=1,dtype=np.float),np.sum(counts,axis=1)))

def histmean(bins,counts):
    return moment(bins,counts,1)

def histrms(bins,counts):
    return np.power(moment(bins,counts,2)-np.power(moment(bins,counts,1),2),0.5)
    
def moments_correlation_vector(moments,index):
    # calculate moment distance of hist at index wrt all other hists
    # very similar to mse_correlation_vector but using histogram moments instead of full histograms for speed-up
    return mse_correlation_vector(moments,index)

def goodnoise(nbins,fstd=None):
    # generate one sample of 'good' noise consisting of fourier components
    # input args:
    # - nbins: number of bins, length of noise array to be sampled
    # - fstd: an array of length nbins used for scaling of the amplitude of the noise
    #         bin-by-bin.
    # output: 
    # - numpy array of length nbins containing the noise
    kmaxscale = 0.25 # frequency limiting factor to ensure smoothness
    ncomps = 3 # number of random sines to use
    kmax = np.pi*kmaxscale
    xax = np.arange(0,nbins)
    noise = np.zeros(nbins)
    # get uniformly sampled wavenumbers in range (0,kmax)
    k = np.random.uniform(low=0,high=1,size=ncomps)*kmax
    # get uniformly sampled phases in range (0,2pi)
    phase = np.random.uniform(low=0,high=1,size=ncomps)*2*np.pi
    # get uniformly sampled amplitudes in range (0,2/ncomps) (i.e. mean total amplitude = 1)
    amplitude = np.random.uniform(low=0,high=1,size=ncomps)*2/ncomps
    for i in range(ncomps):
        temp = amplitude[i]*np.sin(k[i]*xax + phase[i])
        if fstd is not None: temp = np.multiply(temp,fstd)
        noise += temp
    return noise

def badnoise(nbins,fstd=None):
    # generate one sample of 'bad' noise consisting of fourier components
    # (higher frequency and amplitude than 'good' noise)
    # input args and output: simlar to goodnoise
    # WARNING: this noise is not representative of the 'bad' histograms to be expected!
    ampscale = 10. # additional amplitude scaling
    kmaxscale = 1. # additional scaling of max frequency
    kminoffset = 0.5 # additional scaling of min frequency
    ncomps = 3 # number of fourier components
    kmax = np.pi*kmaxscale
    xax = np.arange(0,nbins)
    noise = np.zeros(nbins)
    # get uniformly sampled wavenumbers in range (kmin,kmax)
    k = np.random.uniform(low=kminoffset,high=1,size=ncomps)*kmax
    # get uniformly sampled phases in range (0,2pi)
    phase = np.random.uniform(low=0,high=1,size=ncomps)*2*np.pi
    # get uniformly sampled amplitudes in range (0,2*ampscale/ncomps) (i.e. mean total amplitude = ampscale)
    amplitude = ampscale*np.random.uniform(low=0,high=1,size=ncomps)*2/ncomps
    for i in range(ncomps):
        temp = amplitude[i]*np.sin(k[i]*xax + phase[i])
        if fstd is not None: temp = np.multiply(temp,fstd)
        noise += temp
    return noise

def whitenoise(nbins,fstd=None):
    # generate one sample of white noise (uncorrelated between bins)
    # input args and output: similar to goodnoise
    noise = np.random.normal(size=nbins)
    if fstd is not None: noise = np.multiply(noise,fstd)
    return noise

def random_lico(hists):
    # generate one linear combination of histograms with random coefficients in (0,1) summing to 1
    # input args: 
    # - numpy array of shape (nhists,nbins), the rows of which will be linearly combined
    # output:
    # - numpy array of shape (nbins), containing the new histogram
    nhists = hists.shape[0]
    coeffs = np.random.uniform(low=0.,high=1.,size=nhists)
    coeffs = coeffs/np.sum(coeffs)
    res = np.sum(hists*coeffs[:,np.newaxis],axis=0)
    return res

def mse_correlation_vector(hists,index):
    # calculate mse of a histogram at given index wrt all other histograms
    # input args:
    # - hists: numpy array of shape (nhists,nbins) containing the histograms
    # - index: the index (must be in (0,len(hists)-1)) of the histogram in question
    # output:
    # - numpy array of length nhists containing mse of the indexed histogram with respect to all other histograms
    # WARNING: can be slow if called many times on a large collection of histograms with many bins.
    corvec = np.zeros(len(hists))
    temp = hists - np.tile(hists[index:index+1],(len(hists),1))
    temp = np.power(temp,2)
    corvec = np.mean(temp,axis=1)
    return corvec

def gen_Nshifts( input_hist, n_copies ):
   # for input_hist ogram generate n_copies which with random noise
   # for each bin i calculate random migrations with rate (should be values in range 0.00 to 1.00)

   fluct = np.asarray( [ np.asarray([0.]*len(input_hist)) for _ in range(n_copies)])
   for i in range(n_copies):
      fluct[i] = [rn.gauss(0,1) for i in range(len(input_hist))]

   # Multiply each modification histogram with sqrt of original histogram -> to give same scale to modifications, 
   # so that the modification is applied proportionatelly to each bin. Add to original histogram to get the set of modified ones.

   hist_sqrt = np.sqrt( input_hist )
   fluct = np.asarray( [ input_hist+fluct[i]*hist_sqrt for i in range( n_copies ) ])

   return fluct

def gen_migrations( input_hist, n_copies, rate ):
   # for input_hist ogram generate n_copies which with random migrations between the bins
   # rate - rate of migrations -> generated with normal distribution with std dev = rate, multiplied by bin content
   # for each bin i calculate random migrations with rate (should be values in range 0.00 to 1.00)
   length = len(input_hist)
   output  = np.asarray( [ input_hist for _ in range(n_copies)])
   for i in range(n_copies):
      migr_up = np.asarray( [rn.gauss(0,rate) for _ in range(length)] )
      migr_dw = np.asarray( [rn.gauss(0,rate) for _ in range(length)] )
      temp = np.asarray([0.]*length)
      for j in range(length):
         if ( (j > 0) and (j<length-1) ): 
            temp[j] = input_hist[j-1] * migr_up[j-1] + input_hist[j+1] * migr_dw[j+1]
         elif (j == 0) :
            temp[j] = input_hist[j+1] * migr_dw[j+1]
         elif (j == length-1) :
            temp[j] = input_hist[j-1] * migr_up[j-1]
      output[i] = output[i] - input_hist*( migr_up + migr_dw ) + temp
   return output
      
def migrations(ihists, quantity, rate):
    
   (nhists,nbins) = ihists.shape
   output = ihists
   for i in range(nhists):
      temp = gen_migrations(ihists[i], quantity, rate )
      output = np.concatenate((output,temp))
   return output
