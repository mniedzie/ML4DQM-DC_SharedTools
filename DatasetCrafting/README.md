# ML4DQM tools to generate new histograms
Code to generate new histograms based on random histograms chosen from the input files. Eventually it will generate based on a file with input histograms.



## Usage of the tool

* python3 run_file.py -i data/random_subset_MainDiagonal.txt -o test.csv

where *data/random_subset_MainDiagonal.txt* is an example input file that is already in the repository. *test.csv* Is the example output file.
In current form the code takes 50 randomly chosen histograms from the input file 
cks the ones from Golden Json, and generates 
* 10 histograms using linear combinations of similar histograms
* 20 histograms using Fourier noise
* double the amount of the initial histograms using migrations

These numbers are fully arbitrary, and the three modifications are applied independently, not stacked. The output csv file is generated with all histograms saved as arrays.
