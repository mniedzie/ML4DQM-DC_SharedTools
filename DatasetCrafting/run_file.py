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

#from resamplers import *
from data_handling import *
from generators import *
from plotters import *

import sys, getopt
from sklearn.preprocessing import normalize



if __name__ == '__main__' :

   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print('Input file is "', inputfile ,'"')
   print('Output file is "', outputfile ,'"')

   # read in the csv file, and if Json and DSC info is not there, insert it.
   # Warning: the Golden and Bad json files are hard coded in the function!

   df = read_csv(inputfile, 50)
   print('Input dataframe shape: ', df.shape )

   # I keep the histogtrams for golden (run,ls), can be omitted/modified to keep DSCon
   df = df.loc[ df['Json'] == True ]
   print('Input golden dataframe shape: ', df.shape )


   (hist,runnbs,lsnbs) = get_hist_values(df)

   hist = hist[:,1:-1]
   hist = normalize(hist, norm='l1', axis=1) #normalise the sample, i.e the rows

   ghist = resample_similar_lico( hist, nresamples=10, nonnegative=True, keeppercentage=0.1,whitenoisefactor=0.0)
   fhist = resample_similar_fourier_noise( hist, nresamples=20, nonnegative=False, keeppercentage=1.,whitenoisefactor=0.)
   mhist = migrations(hist, 2, 0.05)

   print("Used ", hist.shape[0], " histograms to generate: \n", ghist.shape[0], " new histograms using linear combinations of similar histograms, \n", mhist.shape[0], " histograms using migrations, \n", fhist.shape[0], " histograms using fourier noise\n")

   output = np.concatenate((mhist, ghist, fhist))
   np.savetxt(outputfile, output, delimiter=",")
