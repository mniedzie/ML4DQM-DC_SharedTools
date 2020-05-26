# ML4DQM tools to generate new histograms
Code to generate new histograms based on the histograms provided and the full set of available histograms. 


## Usage of the tool

Example of the usage of the code:

```
python3 run_file.py -s data/random_subset_NumberOfClustersInStrip_1hist_ex3.txt -i data/ZeroBias_Full2017_DataFrame_1D_Sorted_NumberOfClustersInStrip.txt -r resample_similar_lico --noise white_noise --nresamples 10 --figname=output/NumberOfClustersInStrip_ex3_resample_similar_lico_white_noise.png
```

* -s or --seed defines the file with input histograms meant to be resampled. If seed file is not provided, a 10 random histograms will be chosen from the input file
* -i or --infile defines the input file, with a set of same type of histograms as seed file. Preferably, this file should contain all available histograms of given type. These histograms are used to find a set of similar ones to the resampled histogram.
* -o or --outfile deifnes the output csv file with resampled histograms. If not defined, some default filename is used.
* -r or --resampling defines the resampling algorithm to be used. Available options are the following
  * resample_similar_fourier_noise
  * resample_similar_lico
  * resample_similar_bin_per_bin
  * mc_sampling
* --noise defines what noise methods will be used. Available options are:
  * white_noise
  * fourier_noise
  * migrations
* --nresamples defines how many histograms will be resampled from each input histogram.
* --figname defines the base of the output figure. Be default some plots are printed, including the plots with resampled histograms with the seed histogram.

Description of the algorithms can be found in the following [presentation](https://indico.cern.ch/event/921028/contributions/3869615/attachments/2043283/3422582/presentation.pdf)


Planned update: choice of seed histograms based on run number and lumi section.
