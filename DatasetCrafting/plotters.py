import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # import colormaps

def plotModified(reference, histo, color = 1 ):
    
    # plot original histogram (reference) with ones resampled from it (histo), 
    # histo is either an array of histograms or a single histogram.
    # reference is a dataframe element.
    if color == 1:
       colors = [cm.viridis(i) for i in np.linspace(0, 1, len( histo ))]
    elif color == 2:
       colors = [cm.plasma(i) for i in np.linspace(0, 1, len( histo ))]
    else :
       colors = [cm.inferno(i) for i in np.linspace(0, 1, len( histo ))]
    name=reference['hname']
    vmin=reference['Xmin']
    vmax=reference['Xmax']
    
    # create an array with x values, from x_min to x_max and length as input histogram
    x= vmin + (np.arange(len(histo[0]))) * float((vmax-vmin)/float(len(histo[0])))
    x = x[1:-1]

    for i in range(len(histo)):
       plt.xlim(vmin,vmax)
       
       histo_todraw = histo[i][1:-1]
  
       #srun=df_plot.index.get_level_values('fromrun')[num]
       #slumi=df_plot.index.get_level_values('fromlumi')[num]
       
       plt.step(x, histo_todraw, where='mid', label=(name + " LS " + str(reference['fromlumi']) + " Run " + str(reference['fromrun']) ), color=colors[i])

def plotOrig(reference ):
    
    # reference is a dataframe element.
    histo=reference['histo']
    name=reference['hname']
    vmin=reference['Xmin']
    vmax=reference['Xmax']
    
    # create an array with x values, from x_min to x_max and length as input histogram
    x= vmin + (np.arange(len(histo))) * float((vmax-vmin)/float(len(histo)))
    plt.xlim(vmin,vmax)
    
    x = x[1:-1]
    histo = histo[1:-1]

    #srun=df_plot.index.get_level_values('fromrun')[num]
    #slumi=df_plot.index.get_level_values('fromlumi')[num]
    
    plt.step(x, histo, where='mid', label=(name + " LS " + str(reference['fromlumi']) + " Run " + str(reference['fromrun']) ), color='red')

### plot functions

def plot_data_good_bad(nplot,rhist,ghist,bhist,figname='fig.png'):
    # plot a couple of random examples from rhist (data), ghist (resampled 'good') and bhist (resampled 'bad')
    # input:
    # - nplot: integer, number of examples to plot
    # - rhist, ghist, bhist: numpy arrays of shape (nhists,nbins)
    # - figname: name of figure to plot
    # REPLACED BY PLOT_DATA_AND_GEN, DO NOT USE ANYMORE

    # make sure that figname contains absolute path
    figname = os.path.abspath(figname)

    # data
    randint = np.random.choice(np.arange(len(rhist)),size=min(len(rhist),nplot),replace=False)
    plt.figure()
    for i in randint: plt.plot(rhist[i,:],color='r')
    plt.title('histograms from real data')
    plt.savefig(figname.split('.')[0]+'_data.png')
    # artificial good histograms
    randint = np.random.choice(np.arange(len(ghist)),size=min(len(ghist),nplot),replace=False)
    plt.figure()
    for i in randint: plt.plot(ghist[int(i),:],color='b')
    plt.title('artificial good histograms')
    plt.savefig(figname.split('.')[0]+'_good.png')
    # artificial bad histograms
    randint = np.random.choice(np.arange(len(bhist)),size=min(len(bhist),nplot),replace=False)
    plt.figure()
    for i in randint[:10]: plt.plot(bhist[int(i),:],color='b')
    plt.title('artificial bad histograms')
    plt.savefig(figname.split('.')[0]+'_bad.png')

def plot_data_and_gen(nplot,datahist,genhist,figname='fig.png'):
    # plot a couple of random examples from rhist (data), ghist (resampled 'good') and bhist (resampled 'bad')
    # input:
    # - nplot: integer, maximum number of examples to plot
    # - datahist, genhist: numpy arrays of shape (nhists,nbins)
    # - figname: name of figure to plot

    # make sure that figname contains absolute path
    figname = os.path.abspath(figname)

    # data
    randint = np.random.choice(np.arange(len(datahist)),size=min(len(datahist),nplot),replace=False)
    plt.figure()
    for i in randint: plt.plot(datahist[i,:],color='r')
    plt.title('histograms from data')
    plt.savefig(figname.split('.')[0]+'_data.png')
    # artificial histograms
    randint = np.random.choice(np.arange(len(genhist)),size=min(len(genhist),nplot),replace=False)
    plt.figure()
    for i in randint: plt.plot(genhist[int(i),:],color='b')
    plt.title('artificially generated histograms')
    plt.savefig(figname.split('.')[0]+'_gen.png')


def plot_seed_and_gen(seedhist,genhist,figname='fig.png'):
    # plot a couple of random examples from rhist (data), ghist (resampled 'good') and bhist (resampled 'bad')
    # input:
    # - datahist, genhist: numpy arrays of shape (nhists,nbins)
    # - figname: name of figure to plot

    # make sure that figname contains absolute path
    figname = os.path.abspath(figname)

    # data
    plt.figure()
    gen_colors = [cm.viridis(i) for i in np.linspace(0, 1, len( genhist ))]
    seed_colors = [cm.Reds(i) for i in np.linspace(0, 1, len( seedhist ))]
    for i in range(len(genhist)): plt.plot(genhist[i,:], color = gen_colors[i] )
    for i in range(len(seedhist)): plt.plot(seedhist[i,:], color = 'r',label='seed')
    plt.title('seed and resampled histograms')
    plt.legend()
    plt.savefig(figname.split('.')[0]+'_show.png')


def plot_noise(noise,histstd=None,figname='fig.png'):
    # plot histograms in noise (numpy array of shape (nhists,nbins))
    # optional argument histstd plots +- histstd as boundaries

    # make sure that figname contains absolute path
    figname = os.path.abspath(figname)

    plt.figure()
    for i in range(len(noise)): plt.plot(noise[i,:],'r--')
    if histstd is not None:
        plt.plot(histstd,'k--',label='pm 1 std')
        plt.plot(-histstd,'k--')
    plt.legend()
    plt.title('examples of noise')
    plt.savefig(figname.split('.')[0]+'_noise.png')
