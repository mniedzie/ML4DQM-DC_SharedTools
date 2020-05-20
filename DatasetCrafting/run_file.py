#!/usr/bin/env python

################################################################################
# created by Luka Lambrecht and Marek Niedziela                                #
# a set of functions to process dataframes used for ML4DQM working group       #
# 8th May, 2020                                                                #
################################################################################

import sys, getopt
import matplotlib.pyplot as plt
from matplotlib import cm # import colormaps
import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.preprocessing import normalize

# all the libraries for the model

import math
import random as rn

from data_handling import *
from generators import *
from plotters import *

import sys, getopt
from sklearn.preprocessing import normalize



if __name__ == '__main__' :

    inputfile = ''
    outputfile = ''
    seedfile = ''
    resamplingmethod = 'resample_bin_per_bin'
    noisemethod = '' # default empty string means no noise addition
    nresamples = 10
    figname = '' # default empty string means no plotting
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:s:r:n",
                                   ["help","infile=","outfile=","seedfile=","resampling=","noise=",
                                   "nresamples=",'figname='])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -s <seedfile> -o <outputfile> -r <resampling method> -n <noise method>')
        sys.exit(2)
    for opt, arg in opts:
        print(opt,arg)
        if opt in ("-h", "--help"):
            print('test.py -i <inputfile> -s <seedfile> -o <outputfile> -r <resampling method> -n <noise method>')
            sys.exit()
        if opt in ("-i", "--infile"):
            inputfile = arg
        if opt in ("-o", "--outfile"):
             outputfile = arg
        if opt in ("-s", "--seedfile"):
             seedfile = arg
        if opt in ("-r", "--resampling"):
             resamplingmethod = arg
        if opt in ("-n", "--noise"):
             noisemethod = arg
        if opt in ("--nresamples"):
             nresamples = int(arg)
        if opt in ("--figname"):
             figname = arg
    print('Input file is        : "', inputfile ,'"')
    print('Output file is       : "', outputfile ,'"')
    print('Figure saved to      : "', figname ,'"')
    print('Seed file is         : "', seedfile ,'"')
    print('Resampling method is : "', resamplingmethod ,'"')
    print('Noise method is      : "', noisemethod ,'"')
    print('Number of samples is :  '+str(nresamples))

    # create output folders and make paths absolute
    if not os.path.exists(os.path.dirname(outputfile)) and outputfile!='':
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        outputfile = os.path.abspath(outputfile)
    elif outputfile=='':
        print('no outputfile defined, will use output/test.csv')
        outputfile = 'output/test.csv'
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        outputfile = os.path.abspath(outputfile)
    if not os.path.exists(os.path.dirname(figname)) and figname!='':
        os.makedirs(os.path.dirname(figname), exist_ok=True)
        figname = os.path.abspath(figname)
    if not os.path.exists(os.path.dirname('seed_'+figname)) and figname!='':
        os.makedirs(os.path.dirname('seed_'+figname), exist_ok=True)

    # read in the csv file, and if Json and DSC info is not there, insert it.
    # Warning: the Golden and Bad json files are hard coded in the function!
    df = read_csv(inputfile, -1)
    # uncomment this line to keep only histograms in golden json (if not filtered before)
    #df = df.loc[ df['Json'] == True ]

    (hist,runnbs,lsnbs) = get_hist_values(df)
    # remove under- and overflow bins:
    #hist = hist[:,1:-1]
    # normalize the histograms:
    #hist = normalize(hist, norm='l1', axis=1) #normalise the sample, i.e the rows
  
    # read in the seed file. If not provided, get 10 random histograms from the input file
    if seedfile!='':
        df_seed = read_csv(seedfile, -1)
    else:
        print('No seed file provided, taking random 10 histograms as seed')
        df_seed = read_csv(inputfile, 10)
    # uncomment this line to keep only histograms in golden json (if not filtered before)
    #df_seed = df_seed.loc[ df_seed['Json'] == True ]
    (hist_seed,runnbs_seed,lsnbs_seed) = get_hist_values(df_seed)
 
    resampled_hists = np.zeros(( nresamples, hist.shape[1]))
    smeared_hists = np.zeros(( nresamples, hist.shape[1]))
    if resamplingmethod == 'resample_similar_fourier_noise':
        resampled_hists = resample_similar_fourier_noise(hist,hist_seed,nresamples=nresamples)
    elif resamplingmethod == 'resample_similar_lico':
        resampled_hists = resample_similar_lico(hist,hist_seed,nresamples=nresamples)
    elif resamplingmethod == 'resample_bin_per_bin':
        resampled_hists = resample_bin_per_bin(hist,nresamples=nresamples)
    elif resamplingmethod == 'mc_sampling':
        resampled_hists = mc_sampling(hist_seed,nresamples=nresamples)
     # to add: MC method
    else:
        print('### ERROR ###: resampling method "'+resamplingmethod+'" not recognized.')
        sys.exit()
    if noisemethod == '':
        smeared_hists = resampled_hists
    elif noisemethod == 'white_noise':
        smeared_hists = white_noise(resampled_hists)
    elif noisemethod == 'fourier_noise':
        smeared_hists = fourier_noise(resampled_hists)
    elif noisemethod == 'migrations':
        smeared_hists = migrations(resampled_hists)
    else:
        print('### ERROR ###: noise method "'+noisemethod+'" not recognized.')
        sys.exit()
    outputfile = outputfile.split('.')[0]+'.csv'
    np.savetxt(outputfile,smeared_hists)
    print('Output written to '+outputfile)
    if figname != '': 
        plot_data_and_gen(50,hist,smeared_hists,figname=figname)
        plot_data_and_gen(1,hist_seed,hist_seed,figname='seed_'+figname)
        print('Output plots written to '+figname)
        sys.exit()

