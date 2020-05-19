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

def read_csv(csv_file, rn_samples=-1):
    #### Read dataset. If it does not contain info on GoldenJson and DCSbit, add it
    if not os.path.exists(csv_file):
        print('requested csv file '+csv_file+' does not seem to exist...')
    df = pd.read_csv(csv_file)
    # I throw out empty histograms
    df = df.loc[ df['entries'] != 0 ]
    df['histo'] = df['histo'].apply(literal_eval) # convert histo (str) into a list
    GoldenJson17 = json.load(open('data/GoldenJSON17.json'))
    BadJson17 = json.load(open('data/JsonBAD17.json'))
    df.set_index(['fromrun','fromlumi'], inplace=True, drop=False)
    df.sort_index(inplace=True)
    if not 'Json' in df:
       df["Json"]=False
       for run in df['fromrun'].unique():
          for ls in df['fromlumi'][run]:
             df['Json'][run][ls] = isInJson( run, ls, GoldenJson17)
    if not 'DCSbit' in df:
       df["DCSbit"]=False
       for run in df['fromrun'].unique():
          for ls in df['fromlumi'][run]:
             df['DCSbit'][run][ls] = isDCSon( run, ls, GoldenJson17, BadJson17)
    # select a random subset of the dataframe 
    if(rn_samples!=-1):
       df = df.loc[rn.sample(list(df.index), rn_samples)]
    return df

def isInJson( run, ls, Json):
    #### check if run number 'run' and lumi section 'ls' is in  

    flag = False

    if str(run) in Json.keys():   # check if given run is in json at all
       for i in Json[str(run)]:   # if is, run over lumi sections
          if ( ls >= i[0] and ls <=i[1] ):
             flag = True
             return flag
    return flag
         
def isDCSon( run, ls, GoldenJson, BadJson):
    ### check if run, ls is in one of the input jsons
    return (isInJson(run,ls, GoldenJson) or isInJson(run,ls, BadJson))

# functions to obtain histograms in np array format

def get_hist_values(df):
    ### same as builtin "df['histo'].values" but convert strings to np arrays
    # also an array of run and LS numbers is returned (NEW)
    # warning: no check is done to assure that all histograms are of the same type!

    nn = len(df.loc[df.index[0]]['histo'])
    vals = np.zeros((len(df),nn))
    ls = np.zeros(len(df))
    runs = np.zeros(len(df))
    i = 0
    for r,l in list(df.index):
       one_point = df.loc[r,l]
       vals[i,:] = np.asarray(one_point['histo'])
       ls[i] = r
       runs[i] = l
       i+=1
    return (vals,runs,ls)
         
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
