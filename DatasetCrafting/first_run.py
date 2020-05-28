#!/usr/bin/env python

################################################################################
# created by Luka Lambrecht and Marek Niedziela                                #
# the sript for a first test run of the code to pick the right resampling      #
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
    figname = ''
    nresamples = 10
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:s",
                                   ["help","infile=","outfile=","seedfile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -s <seedfile> -o <outputfile> ')
        sys.exit(2)
    for opt, arg in opts:
        print(opt,arg)
        if opt in ("-h", "--help"):
            print('test.py -i <inputfile> -s <seedfile> -o <outputfile>')
            sys.exit()
        if opt in ("-i", "--infile"):
            inputfile = arg
        if opt in ("-o", "--outfile"):
             outputfile = arg
        if opt in ("-s", "--seedfile"):
             seedfile = arg
    print('Input file is        : "', inputfile ,'"')
    print('Output file is       : "', outputfile ,'"')
    print('Seed file is         : "', seedfile ,'"')

    # create output folders and make paths absolute
    if not os.path.exists(os.path.dirname(outputfile)) and outputfile!='':
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        outputfile = os.path.abspath(outputfile)
    elif outputfile=='':
        print('no outputfile defined, will use output/test.csv')
        outputfile = 'first_run/test.csv'
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        outputfile = os.path.abspath(outputfile)
    if not os.path.exists('first_run'):
        os.makedirs('first_run', exist_ok=True)

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
    print(hist_seed.shape)
 
    # apply each of the resampling methods on the seed histograms.

    hists_sfn  = resample_similar_fourier_noise(hist,hist_seed,nresamples=nresamples, figname=figname)
    hists_sl   = resample_similar_lico(hist,hist_seed,nresamples=nresamples, figname=figname)
    hists_sbpb = resample_similar_bin_per_bin(hist,hist_seed,nresamples=nresamples, figname=figname)
    hists_mc   = mc_sampling(hist_seed,nresamples=nresamples)

    outputfile_sfn  = outputfile.split('.')[0]+'_sfn.csv'
    outputfile_sl   = outputfile.split('.')[0]+'_sl.csv'
    outputfile_sbpb = outputfile.split('.')[0]+'_sbpb.csv'
    outputfile_mc   = outputfile.split('.')[0]+'_mc.csv'

    np.savetxt(outputfile_sfn ,hists_sfn )
    np.savetxt(outputfile_sl  ,hists_sl  )
    np.savetxt(outputfile_sbpb,hists_sbpb)
    np.savetxt(outputfile_mc  ,hists_mc  )

    plot_seed_and_gen( hist_seed, hists_sfn ,               figname='first_run/test_sfn' )
    plot_seed_and_gen( hist_seed, white_noise(hists_sfn),   figname='first_run/test_sfn_wn' )
    plot_seed_and_gen( hist_seed, fourier_noise(hists_sfn), figname='first_run/test_sfn_fn' )
    plot_seed_and_gen( hist_seed, migrations(hists_sfn),    figname='first_run/test_sfn_mi' )

    plot_seed_and_gen( hist_seed, hists_sl,                figname='first_run/test_sl' )
    plot_seed_and_gen( hist_seed, white_noise(hists_sl),   figname='first_run/test_sl_wn' )
    plot_seed_and_gen( hist_seed, fourier_noise(hists_sl), figname='first_run/test_sl_fn' )
    plot_seed_and_gen( hist_seed, migrations(hists_sl),    figname='first_run/test_sl_mi' )

    plot_seed_and_gen( hist_seed, hists_sbpb,                figname='first_run/test_sbpb')
    plot_seed_and_gen( hist_seed, white_noise(hists_sbpb),   figname='first_run/test_sbpb_wn')
    plot_seed_and_gen( hist_seed, fourier_noise(hists_sbpb), figname='first_run/test_sbpb_fn')
    plot_seed_and_gen( hist_seed, migrations(hists_sbpb),    figname='first_run/test_sbpb_mi')
    
    plot_seed_and_gen( hist_seed, hists_mc,                figname='first_run/test_mc' )
    plot_seed_and_gen( hist_seed, white_noise(hists_mc),   figname='first_run/test_mc_wn' )
    plot_seed_and_gen( hist_seed, fourier_noise(hists_mc), figname='first_run/test_mc_fn' )
    plot_seed_and_gen( hist_seed, migrations(hists_mc),    figname='first_run/test_mc_mi' )
