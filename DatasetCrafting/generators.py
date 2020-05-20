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

##### RESAMPLING METHODS #####

def resample_similar_lico( allhists, selhists, outfilename='', figname='', nresamples=1, nonnegative=True,
                          keeppercentage=1.):
    # take linear combinations of similar histograms
    # input arguments:
    # - allhists: 2D np array (nhists,nbins) with all available histograms, used to take linear combinations
    # - selhists: 2D np array (nhists,nbins) with selected hists used for seeding (e.g. 'good' histograms)
    # - outfilename: path to csv file to write result to (default: no writing)
    # - figname: path to figure plotting examples (defautl: no plotting)
    # - nresamples: number of combinations to make per input histogram
    # - nonnegative: boolean whether to make all final histograms nonnegative
    # - keeppercentage: percentage (between 0. and 100.) of histograms in allhists to use per input histogram

    # advantages: no assumptions on noise
    # disadvantages: sensitive to outlying histograms (more than with averaging)
    
    # get some parameters
    if(len(allhists.shape)!=len(selhists.shape) or allhists.shape[1]!=selhists.shape[1]):
        print('### ERROR ###: shapes of allhists and selhists not compatible.')
        return
    (nhists,nbins) = allhists.shape
    (nsel,_) = selhists.shape
    
    # get array of moments (used to define similar histograms)
    binwidth = 1./nbins
    bincenters = np.linspace(binwidth/2,1-binwidth/2,num=nbins,endpoint=True)
    orders = [0,1,2]
    allmoments = np.zeros((nhists,len(orders)))
    for i,j in enumerate(orders): allmoments[:,i] = moment(bincenters,allhists,j)
    selmoments = np.zeros((nsel,len(orders)))
    for i,j in enumerate(orders): selmoments[:,i] = moment(bincenters,selhists,j)
    
    # make resampled histograms
    reshists = np.zeros((nsel*nresamples,nbins))
    for i in range(nsel):
        # select similar histograms
        thisdiff = moments_correlation_vector(np.vstack((selmoments[i],allmoments)),0)[1:]
        #thisdiff = mse_correlation_vector(np.vstack((selhists[i],allhists)),0)[1:]
        threshold = np.percentile(thisdiff,keeppercentage)
        simindices = np.nonzero(np.where(thisdiff<threshold,1,0))[0]
    for j in range(nresamples):
        reshists[nresamples*i+j,:] = random_lico(allhists[simindices,:])
    if nonnegative: reshists = np.maximum(0,reshists)
    np.random.shuffle(reshists)
    nsim = len(simindices)
    print('Note: linear combination is taken between '+str(nsim)+' histograms.')
    print('If this number is too low, histograms might be too similar for combination to have effect.')
    print('If this number is too high, systematic shifts of histogram shapes are included into the combination')
        
    # plot examples of good and bad histograms
    # use only those histograms from real data that were used to create the resamples
    if len(figname)>0: plot_data_and_gen(50,selhists,reshists,figname)
        
    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)

    return reshists


def resample_similar_fourier_noise( allhists, selhists, outfilename='', figname='', nresamples=0, nonnegative=True,
                                   keeppercentage=1.):
    # apply fourier noise on mean histogram, 
    # where the mean is determined from a set of similar-looking histograms
    # input args:
    # - allhists: np array (nhists,nbins) containing all available histograms (to determine mean)
    # - selhists: np array (nhists,nbins) conataining selected histograms used as seeds (e.g. 'good' histograms)
    # - outfilename: path of csv file to write results to (default: no writing)
    # - figname: path to figure plotting examples (default: no plotting)
    # - nresamples: number of samples per input histogram in selhists
    # - nonnegative: boolean whether or not to put all bins to minimum zero after applying noise
    # - keeppercentage: percentage (between 1 and 100) of histograms in allhists to use per input histogram

    # advantages: most of fourier_noise_on_mean but can additionally handle shifting histograms,
    #             apart from fourier noise, also white noise can be applied.
    # disadvantages: does not filter out odd histograms as long as enough other odd histograms look more or less similar
    
    # get some parameters
    if(len(allhists.shape)!=len(selhists.shape) or allhists.shape[1]!=selhists.shape[1]):
        print('### ERROR ###: shapes of allhists and selhists not compatible.')
        return
    (nhists,nbins) = allhists.shape
    (nsel,_) = selhists.shape

    # get array of moments (used to define similar histograms)
    binwidth = 1./nbins
    bincenters = np.linspace(binwidth/2,1-binwidth/2,num=nbins,endpoint=True)
    orders = [0,1,2]
    allmoments = np.zeros((nhists,len(orders)))
    for i,j in enumerate(orders): allmoments[:,i] = moment(bincenters,allhists,j)
    selmoments = np.zeros((nsel,len(orders)))
    for i,j in enumerate(orders): selmoments[:,i] = moment(bincenters,selhists,j)
 
    # make resampled histograms
    reshists = np.zeros((nsel*nresamples,nbins))
    for i in range(nsel):
        # select similar histograms
        thisdiff = moments_correlation_vector(np.vstack((selmoments[i],allmoments)),0)[1:]
        #thisdiff = mse_correlation_vector(np.vstack((selhists[i],allhists)),0)[1:]
        threshold = np.percentile(thisdiff,keeppercentage)
        simindices = np.nonzero(np.where(thisdiff<threshold,1,0))[0]
        for j in range(nresamples):
            reshists[nresamples*i+j,:] = fourier_noise_on_mean(allhists[simindices,:],
                                                               figname='',nresamples=1,nonnegative=nonnegative)[0,:]
    if nonnegative: reshists = np.maximum(0,reshists)
    np.random.shuffle(reshists)
    nsim = len(simindices)
    print('Note: mean and std calculation is performed on '+str(nsim)+' histograms.')
    print('If this number is too low, histograms might be too similar for averaging to have effect.')
    print('If this number is too high, systematic shifts of histogram shapes are included into the averaging.')

    # plot examples of good and bad histograms
    # use only those histograms from real data that were used to create the resamples
    if len(figname)>0: plot_data_and_gen(50,selhists,reshists,figname)

    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)

    return reshists

def resample_similar_bin_per_bin( allhists, selhists, outfilename='', figname='', nresamples=1, nonnegative=True,
                                   keeppercentage=1.):
    # resample from bin-per-bin probability distributions, but only from similar looking histograms.
    # input args:
    # - allhists: np array (nhists,nbins) containing all available histograms (to determine mean)
    # - selhists: np array (nhists,nbins) conataining selected histograms used as seeds (e.g. 'good' histograms)
    # - outfilename: path of csv file to write results to (default: no writing)
    # - figname: path to figure plotting examples (default: no plotting)
    # - nresamples: number of samples per input histogram in selhists
    # - nonnegative: boolean whether or not to put all bins to minimum zero after applying noise
    # - keeppercentage: percentage (between 1 and 100) of histograms in allhists to use per input histogram

    # advantages: no assumptions on shape of noise,
    #             can handle systematic shifts in histograms
    # disadvantages: bins are treated independently from each other
 
    # set some parameters
    (nhists,nbins) = allhists.shape
    (nsel,_) = selhists.shape
    
    # get array of moments (used to define similar histograms)
    binwidth = 1./nbins
    bincenters = np.linspace(binwidth/2,1-binwidth/2,num=nbins,endpoint=True)
    orders = [0,1,2]
    moments = np.zeros((nhists,len(orders)))
    for i,j in enumerate(orders): moments[:,i] = moment(bincenters,rhist,j)
    
    # make resamples
    reshists = np.zeros((nsel*nresamples,nbins))
    for i in range(nsel):
        # select similar histograms
        thisdiff = moments_correlation_vector(moments,j)
        #thisdiff = mse_correlation_vector(rhist,j)
        threshold = np.percentile(thisdiff,keeppercentage)
        simindices = np.nonzero(np.where(thisdiff<threshold,1,0))[0]
    for j in range(nresamples):
        reshists[nresamples*i+j,:] = resample_bin_per_bin(allhists[simindices,:],
           figname='',nresamples=1,nonnegative=nonnegative)[0,:]
    if nonnegative: reshists = np.maximum(0,reshists)
    np.random.shuffle(reshists)
    nsim = len(simindices)
    print('Note: bin-per-bin resampling performed on '+str(nsim)+' histograms.')
    print('If this number is too low, existing histograms are drawn with too small variation.')
    print('If this number is too high, systematic shifts of histograms can be averaged out.')
        
    # plot examples of good and bad histograms
    if len(figname)>0: plot_data_and_gen(50,selhists,reshists,figname)

    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)
        
    return reshists

def resample_bin_per_bin(hists,outfilename='',figname='',nresamples=0,nonnegative=True,smoothinghalfwidth=0):
    # do resampling from bin-per-bin probability distributions
    # input args:
    # - hists: np array (nhists,nbins) containing the histograms to draw new samples from
    # - outfilename: path to csv file to write results to (default: no writing)
    # - figname: path to figure plotting examples (default: no plotting)
    # - nresamples: number of samples to draw (default: 1/10 of number of input histograms)
    # - nonnegative: boolean whether or not to put all bins to minimum zero after applying noise
    # - smoothinghalfwidth: halfwidth of smoothing procedure to apply on the result (default: no smoothing)

    # advantages: no arbitrary noise modeling
    # disadvantages: bins are considered independent, shape of historams not taken into account,
    #                does not work well on small number of input histograms, 
    #                does not work well on histograms with systematic shifts
    
    if nresamples==0: nresamples=int(len(hists)/10)
    nbins = hists.shape[1]
    
    # generate data
    reshists = np.zeros((nresamples,nbins))
    for i in range(nbins):
        col = np.random.choice(hists[:,i],size=nresamples,replace=True)
        reshists[:,i] = col
        
    # apply smoothing to compensate partially for bin independence
    if smoothinghalfwidth>0: reshists = smoother(reshists,halfwidth=smoothinghalfwidth)

    # plot examples of good and bad histograms
    if len(figname)>0: plot_data_and_gen(50,hists,reshists,figname)
    
    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)

    return reshists


def fourier_noise_on_mean(hists,outfilename='',figname='',nresamples=0,nonnegative=True):
    # apply fourier noise on the bin-per-bin mean histogram,
    # with amplitude scaling based on bin-per-bin std histogram.
    # input args:
    # - hists: numpy array of shape (nhists,nbins) used for determining mean and std
    # - outfilename: path to csv file to write results to (default: no writing)
    # - figname: path to figure plotting examples (default: no plotting)
    # - nresamples: number of samples to draw (default: number of input histograms / 10)
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    # MOSTLY SUITABLE AS HELP FUNCTION FOR RESAMPLE_SIMILAR_FOURIER_NOISE, NOT AS GENERATOR IN ITSELF

    # advantages: mean histogram is almost certainly 'good' because of averaging, eliminate bad histograms
    # disadvantages: deviations from mean are small, does not model systematic shifts by lumi.
    
    if nresamples==0: nresamples=int(len(hists)/10)

    # get mean and std histogram
    histmean = np.mean(hists,axis=0)
    histstd = np.std(hists,axis=0)
    nbins = len(histmean)

    # plot examples of histograms mean, and std
    if len(figname)>0:
        nplot = min(200,len(hists))
        randint = np.random.choice(np.arange(len(hists)),size=nplot,replace=False)
        plt.figure()
        for i in randint: plt.plot(hists[int(i),:],color='b',alpha=0.1)
        plt.plot(histmean,color='black',label='mean')
        plt.plot(histmean-histstd,color='r',label='pm 1 std')
        plt.plot(histmean+histstd,color='r')
        plt.legend()
        plt.savefig(figname.split('.')[0]+'_meanstd.png')
    
    # generate data
    reshists = np.zeros((nresamples,nbins))
    for i in range(nresamples): reshists[i,:] = histmean + goodnoise(nbins,histstd)
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)
        
    # plot examples of good and bad histograms
    if len(figname)>0:
        noise_examples = []
        for i in range(5): noise_examples.append(goodnoise(nbins,histstd))
        plot_noise(np.array(noise_examples),histstd,figname)
        plot_data_and_gen(50,hists,reshists,figname)

    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)
    
    return reshists

def mc_sampling(hists, nMC=10000 , nresamples=10):
    ### resampling of a histogram using MC methods
    # Drawing random points from a space defined by the range of the histogram in all axes.
    # Points are "accepted" if the fall under the sampled histogram:
    # f(x) - sampled distribution
    # x_r, y_r -> randomly sampled point
    # if y_r<=f(x_r), fill the new distribution at bin corresponding to x_r with weight:
    # weight = (sum of input hist)/(#mc points accepted)
    # this is equal to 
    # weight = (MC space volume)/(# all MC points)
    (nHists,nBins) = hists.shape
    output = np.asarray( [ np.asarray([0.]*nBins) for _ in range(nHists*nresamples)])
    for i in range(nHists):
       for j in range(nresamples):
#       norm = np.sum(hists[i])/(nbins*np.max(hists[i]))
          weight = nBins*np.max(hists[i])/nMC
          for _ in range(nMC):
             x_r=rn.randrange(nBins)
             y_r=rn.random()*np.max(hists[i])
             if( y_r <= hists[i][x_r]):
                output[i*nresamples+j][x_r]+=weight
    return output

##### NOISE METHODS #####

def fourier_noise(hists,outfilename='',figname='',nresamples=1,nonnegative=True,stdfactor=15.):
    # apply fourier noise on randomly histograms with simple flat amplitude scaling.
    # input args: 
    # - hists: numpy array of shape (nhists,nbins) used for seeding
    # - outfilename: path to csv file to write results to (default: no writing)
    # - figname: path to figure plotting examples (default: no plotting)
    # - nresamples: number of samples to draw per input histogram
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    # - stdfactor: factor to scale magnitude of noise (larger factor = smaller noise)

    # advantages: resampled histograms will have statistically same features as original input set
    # disadvantages: also 'bad' histograms will be resampled if included in hists
    
    (nhists,nbins) = hists.shape
    
    # generate data
    reshists = np.zeros((nresamples*len(hists),nbins))
    for i in range(nhists):
        for j in range(nresamples):
            reshists[nresamples*i+j,:] = hists[i,:]+goodnoise(nbins,hists[i,:]/stdfactor)
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)
    np.random.shuffle(reshists)

    # plot examples of good and bad histograms
    if len(figname)>0: 
        noise_examples = []
        for i in range(5): noise_examples.append(goodnoise(nbins,hists[-1,:]/stdfactor))
        plot_noise(np.array(noise_examples),hists[-1,:]/stdfactor,figname)
        plot_data_and_gen(50,hists,reshists,figname)
    
    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)

    return reshists

def white_noise(hists,stdfactor=15.):
    # apply white noise to the histograms in hists.
    # input args:
    # - hists: np array (nhists,nbins) containing input histograms
    # - stdfactor: scaling factor of white noise amplitude (higher factor = smaller noise)

    (nhists,nbins) = hists.shape
    print(hists.shape)
    reshists = np.zeros((nhists,nbins))

    for i in range(nhists):
        reshists[i,:] = hists[i,:] + np.multiply( np.random.normal(nbins), hists[i,:]/stdfactor)

    return reshists

def migrations(ihists, quantity=1, rate=0.05):
    # generate migrations between bins of the histograms
    # quantity defines how many resampled copies oh histograms will be made
    # rate limits the migration rate

    (nhists,nbins) = ihists.shape
    output = ihists
    for i in range(nhists):
        temp = gen_migrations(ihists[i], quantity, rate )
        if i==0: output = temp
        else: output = np.concatenate((output,temp))
    return output

##### HELP FUNCTIONS #####

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

def goodnoise( nbins, fstd=None):
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
    # WARNING: NOT NECESSARILY REPRESENTATIVE OF ANOMALIES TO BE EXPECTED, DO NOT USE
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
    # the code takes difference between moment values at position index and moments of all other histograms
    # then squares it and calculates the mean of the (currently three) moments
    corvec = np.zeros(len(hists))
    temp = hists - np.tile(hists[index:index+1],(len(hists),1))
    temp = np.power(temp,2)
    corvec = np.mean(temp,axis=1)
    return corvec

def smoother(inarray,halfwidth):
    # smooth the rows of a 2D array using the 2*halfwidth+1 surrounding values.
    outarray = np.zeros(inarray.shape)
    nbins = inarray.shape[1]
    for j in range(nbins):
        crange = np.arange(max(0,j-halfwidth),min(nbins,j+halfwidth+1))
        outarray[:,j] = np.sum(inarray[:,crange],axis=1)/len(crange)
    return outarray

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
      migr_up = np.asarray( [abs(rn.gauss(0,rate)) for _ in range(length)] )
      migr_dw = np.asarray( [abs(rn.gauss(0,rate)) for _ in range(length)] )
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
