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
    nresamples = 1000
    figname = '' # default empty string means no plotting
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:s:r:n",
                                   ["help","infile=","outfile=","seedfile=","resampling=","noise=",
                                   "nresamples=",'figname='])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -s <seedfile> -o <outputfile> -r <resampling method> -n <noise method>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('test.py -i <inputfile> -s <seedfile> -o <outputfile> -r <resampling method> -n <noise method>')
            sys.exit()
        elif opt in ("-i", "--infile"):
            inputfile = arg
        elif opt in ("-o", "--outfile"):
             outputfile = arg
        elif opt in ("-s", "--seedfile"):
             seedfile = arg
        elif opt in ("-r", "--resampling"):
             resamplingmethod = arg
        elif opt in ("-n", "--noise"):
             noisemethod = arg
        elif opt in ("--nresamples"):
             nresamples = int(arg)
        elif opt in ("--figname"):
             figname = arg
    print('Input file is        : "', inputfile ,'"')
    print('Output file is       : "', outputfile ,'"')
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

    # read in the csv file, and if Json and DSC info is not there, insert it.
    # Warning: the Golden and Bad json files are hard coded in the function!
    df = read_csv(inputfile, -1)
    # uncomment this line to keep only histograms in golden json (if not filtered before)
    #df = df.loc[ df['Json'] == True ]
    print('Input dataframe shape: ', df.shape )

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
        print('Output plots written to '+figname)
        sys.exit()


# The lines below can be used to save the output to a simple csv file. 
#    print("Used ", hist.shape[0], " histograms to generate: \n", ghist.shape[0], " new histograms using linear combinations of similar histograms, \n", mhist.shape[0], " histograms using migrations, \n", fhist.shape[0], " histograms using fourier noise\n")
#
#    output = np.concatenate((mhist, ghist, fhist))
#    np.savetxt(outputfile, output, delimiter=",")






#    colors = [cm.viridis(i) for i in np.linspace(0, 1, len( mcHist ))]
#    for i in range(len(mcHist)):
#        plt.plot(mcHist[i], color=colors[i])
#    plt.plot(hist[0], color='red')
#    plt.title('histograms generated with MC')
#    plt.show()
#
#    ghist = resample_similar_lico( hist, nresamples=10, nonnegative=True, keeppercentage=0.1,whitenoisefactor=0.0)
#    fhist = resample_similar_fourier_noise( hist, nresamples=20, nonnegative=False, keeppercentage=1.,whitenoisefactor=0.)
#    mhist = migrations(hist, 2, 0.05)
#
