#!/usr/bin/env python

################################################################################
# created by Luka Lambrecht and Marek Niedziela                                #
# a set of functions to process dataframes used for ML4DQM working group       #
# 8th May, 2020                                                                #
################################################################################

import sys, getopt
try:
    import configparser
except: 
    import ConfigParser as configparser
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


if __name__ == '__main__' :

    # command line args
    inputfile = ''
    outputfile = ''
    seedfile = ''
    conffile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:s:c:",
                                   ["help","infile=","outfile=","seedfile=","configuration="])
    except getopt.GetoptError:
        print('run_file.py -i <inputfile> -s <seedfile> -o <outputfile> -c <configurationfile>')
        sys.exit(2)
    for opt, arg in opts:
        print(opt,arg)
        if opt in ("-h", "--help"):
            print('run_file.py -i <inputfile> -s <seedfile> -o <outputfile> -c <configurationfile>')
            sys.exit()
        if opt in ("-i", "--infile"):
            inputfile = arg
        if opt in ("-o", "--outfile"):
             outputfile = arg
        if opt in ("-s", "--seedfile"):
             seedfile = arg
        if opt in ("-c", "--configuration"):
             conffile = arg
    if conffile == '':
        print('### ERROR ###: no configuration file was specified, use option -c <configurationfile>.')
        sys.exit(2)
    print('Input file is         : "', inputfile ,'"')
    print('Output file is        : "', outputfile ,'"')
    print('Seed file is          : "', seedfile ,'"')
    print('Configuration file is : "', conffile ,'"')

    # read configuration file
    config = configparser.ConfigParser()
    config.read(conffile)
    resamplingmethod = str(config['core']['resamplingmethod'])
    noisemethod = str(config['core']['noisemethod'])
    if noisemethod == 'none': noisemethod = ''
    nresamples = int(config['core']['nresamples'])
    figname = str(config['core']['figname'])
    if figname == 'none': figname = ''
    resampling_options = {}
    options_key = resamplingmethod+'_options'
    for option in config[options_key]:
        resampling_options[option] = config[options_key][option]
    noise_options = {}
    options_key = noisemethod+'_options'
    if noisemethod != '':
        for option in config[options_key]:
            noise_options[option] = config[options_key][option]

    # create output folders and make paths absolute
    if not os.path.exists(os.path.dirname(outputfile)) and outputfile!='':
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        outputfile = os.path.abspath(outputfile)
    elif outputfile=='':
        print('### WARNING ###: no outputfile defined, will use output/test.csv.')
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
        if seedfile.split('.')[1]=='csv' or seedfile.split('.')[1]=='txt':
            print('Reading in seed from a csv file!')
            df_seed = read_csv(seedfile, -1)
        elif seedfile.split('.')[1]=='json':
            print('Reading in seed json file, will pick seed from input file based on run numbers and lumi sections!')
            df_seed = read_seed_json(df, seedfile)
        else: 
            print('### WARNING ###: WRONG seed file formet, taking random 10 histograms as seed when needed.')
            df_seed = read_csv(inputfile, 10)
    else:
        print('### WARNING ###: no seed file nor json provided, taking random 10 histograms as seed when needed.')
        df_seed = read_csv(inputfile, 10)
    # uncomment this line to keep only histograms in golden json (if not filtered before)
    #df_seed = df_seed.loc[ df_seed['Json'] == True ]
    (hist_seed,runnbs_seed,lsnbs_seed) = get_hist_values(df_seed)
 
    resampled_hists = np.zeros(( nresamples, hist.shape[1]))
    smeared_hists = np.zeros(( nresamples, hist.shape[1]))
    if resamplingmethod == 'resample_similar_fourier_noise':
        resampled_hists = resample_similar_fourier_noise(hist,hist_seed,nresamples=nresamples, 
			    figname=figname,
			    nonnegative=bool(resampling_options['nonnegative']),
			    keeppercentage=float(resampling_options['keeppercentage']))
    elif resamplingmethod == 'resample_similar_lico':
        resampled_hists = resample_similar_lico(hist,hist_seed,nresamples=nresamples, figname=figname,
			    nonnegative=bool(resampling_options['nonnegative']),
			    keeppercentage=float(resampling_options['keeppercentage']))
    elif resamplingmethod == 'resample_similar_bin_per_bin':
        resampled_hists = resample_similar_bin_per_bin(hist,hist_seed,nresamples=nresamples, figname=figname,
			    nonnegative=bool(resampling_options['nonnegative']),
			    keeppercentage=float(resampling_options['keeppercentage']),
			    smoothinghalfwidth=int(resampling_options['smoothinghalfwidth']))
    elif resamplingmethod == 'resample_bin_per_bin':
        resampled_hists = resample_bin_per_bin(hist,nresamples=nresamples, figname=figname,
			    nonnegative=bool(resampling_options['nonnegative']),
			    smoothinghalfwidth=int(resampling_options['smoothinghalfwidth']))
    elif resamplingmethod == 'mc_sampling':
        resampled_hists = mc_sampling(hist_seed,nresamples=nresamples,
			    nMC=int(resampling_options['nmc']))
    else:
        print('### ERROR ###: resampling method "'+resamplingmethod+'" not recognized.')
        sys.exit(2)
    if noisemethod == '':
        smeared_hists = resampled_hists
    elif noisemethod == 'white_noise':
        smeared_hists = white_noise(resampled_hists,
			    nonnegative=bool(noise_options['nonnegative']),
			    stdfactor=float(noise_options['stdfactor']))
    elif noisemethod == 'fourier_noise':
        smeared_hists = fourier_noise(resampled_hists,
			    nonnegative=bool(noise_options['nonnegative']),
			    stdfactor=float(noise_options['stdfactor']))
    elif noisemethod == 'migrations':
        smeared_hists = migrations(resampled_hists,
			    rate=float(noise_options['rate']))
    else:
        print('### ERROR ###: noise method "'+noisemethod+'" not recognized.')
        sys.exit(2)
    outputfile = outputfile.split('.')[0]+'.csv'
    np.savetxt(outputfile,smeared_hists)
    print('Output written to '+outputfile)
    if figname != '': 
        plot_data_and_gen( 50, hist, smeared_hists, figname=figname)
        plot_seed_and_gen( hist_seed, smeared_hists, figname='seed_'+figname)
        print('Output plots written to '+figname)
