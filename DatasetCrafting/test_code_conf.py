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

'''
seed_files = ['data/random_subset_MainDiagonal_1hist_ex1.txt',
              'data/random_subset_MainDiagonal_1hist_ex2.txt',
              'data/random_subset_MainDiagonal_1hist_ex3.txt']
input_files = ['data/random_subset_MainDiagonal.txt',
               'data/random_subset_MainDiagonal.txt',
               'data/random_subset_MainDiagonal.txt']
output_files = ['output/MainDiagonal_ex1',
                'output/MainDiagonal_ex2',
                'output/MainDiagonal_ex3']
'''

'''
seed_files = ['data/random_subset_ChargeInnerLayer4_1hist_ex1.txt',
               'data/random_subset_ChargeInnerLayer4_1hist_ex2.txt',
               'data/random_subset_ChargeInnerLayer4_1hist_ex3.txt',
               'data/random_subset_MainDiagonal_1hist_ex1.txt',
               'data/random_subset_MainDiagonal_1hist_ex2.txt',
               'data/random_subset_MainDiagonal_1hist_ex3.txt',
               'data/random_subset_NumberOfClustersInPixel_1hist_ex1.txt',
               'data/random_subset_NumberOfClustersInPixel_1hist_ex2.txt',
               'data/random_subset_NumberOfClustersInPixel_1hist_ex3.txt',
               'data/random_subset_NumberOfClustersInStrip_1hist_ex1.txt',
               'data/random_subset_NumberOfClustersInStrip_1hist_ex2.txt',
               'data/random_subset_NumberOfClustersInStrip_1hist_ex3.txt']
input_files = ['../../Scripts2020/ZeroBias_2017UL_DataFrame_ChargeInnerLayer4.txt',
               '../../Scripts2020/ZeroBias_2017UL_DataFrame_ChargeInnerLayer4.txt',
               '../../Scripts2020/ZeroBias_2017UL_DataFrame_ChargeInnerLayer4.txt',
               '../../Scripts2020/ZeroBias_Full2017_DataFrame_1D_Sorted_MainDiagonal.txt',
               '../../Scripts2020/ZeroBias_Full2017_DataFrame_1D_Sorted_MainDiagonal.txt',
               '../../Scripts2020/ZeroBias_Full2017_DataFrame_1D_Sorted_MainDiagonal.txt',
               '../../Scripts2020/ZeroBias_Full2017_DataFrame_1D_Sorted_NumberOfClustersInPixel.txt',
               '../../Scripts2020/ZeroBias_Full2017_DataFrame_1D_Sorted_NumberOfClustersInPixel.txt',
               '../../Scripts2020/ZeroBias_Full2017_DataFrame_1D_Sorted_NumberOfClustersInPixel.txt',
               '../../Scripts2020/ZeroBias_Full2017_DataFrame_1D_Sorted_NumberOfClustersInStrip.txt',
               '../../Scripts2020/ZeroBias_Full2017_DataFrame_1D_Sorted_NumberOfClustersInStrip.txt',
               '../../Scripts2020/ZeroBias_Full2017_DataFrame_1D_Sorted_NumberOfClustersInStrip.txt']
output_files = ['output/ChargeInnerLayer4_ex1',
                'output/ChargeInnerLayer4_ex2',
                'output/ChargeInnerLayer4_ex3',
                'output/MainDiagonal_ex1',
                'output/MainDiagonal_ex2',
                'output/MainDiagonal_ex3',
                'output/NumberOfClustersInPixel_ex1',
                'output/NumberOfClustersInPixel_ex2',
                'output/NumberOfClustersInPixel_ex3',
                'output/NumberOfClustersInStrip_ex1',
                'output/NumberOfClustersInStrip_ex2',
                'output/NumberOfClustersInStrip_ex3']
'''

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
            command_run = subprocess.call(command, shell=True)
            if command_run == 0:
                print('Succesfully generated histograms!')
            else:
                print('command failed, will save info to file')
                text_file.write(command+'\n')
text_file.close()
