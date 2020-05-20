#!/usr/bin/env python

################################################################################
# created by Luka Lambrecht and Marek Niedziela                                #
# a set of functions to process dataframes used for ML4DQM working group       #
# 8th May, 2020                                                                #
################################################################################

import os

resamp_methods = [#'resample_similar_fourier_noise',
                  #'resample_similar_lico',
		  'resample_similar_bin_per_bin',
                  #'resample_bin_per_bin',
                  #'mc_sampling'
                  ]
noise_methods = ['',
                 #'white_noise',
                 #'fourier_noise',
                 #'migrations'
                 ]

seed_files = ['data/random_subset_MainDiagonal_1hist_ex1.txt',
              'data/random_subset_MainDiagonal_1hist_ex2.txt',
              'data/random_subset_MainDiagonal_1hist_ex3.txt']
input_files = ['data/random_subset_MainDiagonal.txt',
               'data/random_subset_MainDiagonal.txt',
               'data/random_subset_MainDiagonal.txt']
output_files = ['output/MainDiagonal_ex1',
                'output/MainDiagonal_ex2',
                'output/MainDiagonal_ex3']

for i in range(len(seed_files)):
    for j in resamp_methods:
        for k in noise_methods:
            command = 'python3 run_file.py -s '
            command += seed_files[i] 
            command += ' -i '
            command += input_files[i] 
            command += ' -r '
            command += j
            if k != '': command += ' --noise '
            command += k
            command += ' --nresamples 10 --figname='
            command += output_files[i]
            command += '_'
            command += j
            command += '_'
            command += k
            command += '.png'
            print(command)
            os.system(command)
