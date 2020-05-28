# ML4DQM tools to generate new histograms

Code to generate new histograms based on the histograms provided and the full set of available histograms. 

## Brief introduction

The generates new "resampled" histograms based on a set of provided "seed" histograms. The resampling methods and noise applied on them are described in this [presentation](https://indico.cern.ch/event/921028/contributions/3869615/attachments/2043283/3422582/presentation.pdf). There are four methods of resampling implemented (for more description see the presentation or the comments within the code):

* ```resample_similar_fourier_noise``` - Find similar histograms and apply some smooth noise on it.
* ```resample_similar_lico         ``` - Make a random linear combination of similar histograms.
* ```resample_similar_bin_per_bin  ``` - Randomly draw bin contents from a set of similar histograms.
* ```mc_sampling                   ``` - MC-style sampling.

After resampling, one can applie one of the three noise methods

* ```white_noise``` - a simple white noise, a random shift in each bin drawn from normal distribution
* ```fourier_noise``` - a smooth random noise
* ```migrations``` - simple migrations between bins

## Usage of the tool

### First use - choice of resampling method

For the first run of the code we recommend running the code on a set of example histograms, on which each resampling method will be applied. This can be done using the following script:

```
python3 first_run.py --seedfile <seed_file_path> --infile <input_file_path> --outfile <output_file_path>
```
The input file should contain all available histograms of given type. These are used to find histograms similar to the seed histograms.
The seed file should contain the examples of input histograms (in same dataframe format as used in whole ML4DQM method or as a json file, same format as Golden Json, with picked runs and lumisections). If no file is provided, 10 random histograms from input will be used.
The output file will be used as a basis name to save the outputs (an abbreviation of each method name will be added at the end of the name). By default it is set to ```first_run/test.csv```.

The plots with resampled histograms are saved in ```first_run/``` directory. The naming is "test+resamppling method abbreviation+noise method abbreviation". These are hardcoded, and generally meant as a first step to choose the optimal method. Further optimization of the algorithms should be done within configuration files, using the method explained below.

### Running the production of resampled histograms

Two ways of running the code are implemented. The parameters can be either provided in a configuration file (the recommended procedure), or as a command-line arguments. Both are presented below, first with config file:

```
python3 run_file_conf.py -s <seed_file_path> -i <input_file_path> -c <config_file_path> 
```
* -s or --seedfile defines the file with input histograms meant to be resampled. It can be either csv/txt file which will be loaded into a dataframe, or json file will be used to pick required histograms from input file. If seed file is not provided, a 10 random histograms will be chosen from the input file
* -i or --infile defines the input file, with a set of same type of histograms as seed file. Preferably, this file should contain all available histograms of given type. These histograms are used to find a set of similar ones to the resampled histogram.
* -o or --outfile deifnes the output csv file with resampled histograms. If not defined, some default filename is used.
* -c or --configuration provides path to configuration file. The syntax of configuration file is well explained in ```configurations/example_configuration.cfg```.


Running the code with all parameters provided in command line should look like:
```
python3 run_file.py -s <seed_file> -i <input_file> -r <resample_method> --noise <noise_method> --nresamples <integer> --figname=<figure_name>
```

* -s or --seedfile defines the file with input histograms meant to be resampled. It can be either csv/txt file which will be loaded into a dataframe, or json file will be used to pick required histograms from input file. If seed file is not provided, a 10 random histograms will be chosen from the input file
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
