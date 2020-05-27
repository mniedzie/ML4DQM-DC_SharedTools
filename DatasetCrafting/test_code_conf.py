#!/usr/bin/env python

################################################################################
# created by Luka Lambrecht and Marek Niedziela                                #
# a set of functions to process dataframes used for ML4DQM working group       #
# 8th May, 2020                                                                #
################################################################################

import os
import subprocess

conf_files = ['configurations/ex1.cfg','configurations/ex2.cfg','configurations/ex3.cfg',
		'configurations/ex4.cfg','configurations/ex5.cfg']
seed_files = ['data/random_subset_MainDiagonal_1hist_ex1.txt']
input_files = ['data/random_subset_MainDiagonal.txt']
output_files = ['output/MainDiagonal_ex1.txt']

text_file = open("Error_conf.txt", "w")
for i in range(len(seed_files)):
    for j in conf_files:
            command = 'python3 run_file_conf.py -s '
            command += seed_files[i] 
            command += ' -i '
            command += input_files[i] 
            command += ' -c '
            command += j
            print(command)
#            command_run = subprocess.call(command, shell=True)
#            if command_run == 0:
#                print('Succesfully generated histograms!')
#            else:
#                print('command failed, will save info to file')
#                text_file.write(command+'\n')
text_file.close()
