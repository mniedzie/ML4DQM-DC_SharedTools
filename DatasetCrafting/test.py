###################################
# temporary file for code testing #
###################################
# import python packages
import os
# import local files
import generators as gen
import data_handling as data

inputfile = 'data/random_subset_MainDiagonal.txt'
df = data.read_csv(inputfile, -1)
print('Input dataframe shape: ', df.shape )

df = df.loc[ df['Json'] == True ]
print('Input golden dataframe shape: ', df.shape )

(hist,runnbs,lsnbs) = data.get_hist_values(df)
print('Shape of histogram array: '+str(hist.shape))
selhist = hist[102:103,:]
print('Shape of selected histogram array: '+str(selhist.shape))

outfolder = 'testresults'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

gen.fourier_noise_on_mean(hist,outfilename=os.path.join(outfolder,'fnom'),figname=os.path.join(outfolder,'fnom'),
			    nresamples=10,nonnegative=True)

gen.fourier_noise(selhist,outfilename=os.path.join(outfolder,'fn'),figname=os.path.join(outfolder,'fn'),
			    nresamples=10,nonnegative=True,stdfactor=10)

gen.resample_similar_lico(hist,selhist,outfilename=os.path.join(outfolder,'rsl'),
			    figname=os.path.join(outfolder,'rsl'),
			    nresamples=1,nonnegative=True,keeppercentage=0.05,whitenoisefactor=0.)

gen.resample_similar_fourier_noise(hist,selhist,outfilename=os.path.join(outfolder,'rsfn'),
			    figname=os.path.join(outfolder,'rsfn'),
			    nresamples=10,nonnegative=True,keeppercentage=0.1,whitenoisefactor=0.)

gen.resample_bin_per_bin(hist,outfilename=os.path.join(outfolder,'rbpb'),figname=os.path.join(outfolder,'rbpb'),
			    nresamples=10,nonnegative=True,smoothinghalfwidth=1)

print('success')
