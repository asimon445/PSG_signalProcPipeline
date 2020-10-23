#This will compute delta power changes in the NREM stages throughtout the course of the night. It concatenates all NREM stages and then fits a line to the resulting power vector. Not the ideal way to do this, but it's probably the most reasonable/easy way to do it in this PSP population

from scipy.stats import zscore
from scipy.signal import welch
from scipy.stats import linregress
import yasa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import seaborn as sns
import pandas as pd
sns.set(font_scale=1.2)
import mne
import glob
from scipy.special import erf
import datetime
import csv
import os

PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/preprocessed/'

#create list of PSG files
FILES = glob.glob(PATH + '*.fif')
FILES.sort()

#create list of hypnogram files  
HYPNO = glob.glob(PATH + '*.csv')
HYPNO.sort()

# this is for indexing the last '/' in the path so that we can pull the filename easily for each participant
sep = '/'
def find(PATH, sep):
    return [i for i, ltr in enumerate(PATH) if ltr == sep]

pathsep = list(find(PATH,sep))
pathsep = pathsep[len(pathsep)-1]

sf = 100.

#format output header  -- slow wave density
Stats = np.zeros((len(FILES)+1,2), dtype=np.int)
Stats = Stats.astype('U20')

Stats[0,0] = 'Subject'
Stats[0,1] = 'Delta degradation slope'

for f, file in enumerate(FILES):

    PSG_fnum = FILES[f][np.s_[pathsep+1:pathsep+6]]
    Hyp_fnum= HYPNO[f][np.s_[pathsep+1:pathsep+6]]

    # make sure that the participant numbers between the PSG file and the hypnogram file for iteration 'f' are the same
    if PSG_fnum == Hyp_fnum:

        eeg = mne.io.read_raw_fif(FILES[f], preload=True)
        data = eeg.get_data() 
        data = data*1000000

        channels = eeg.ch_names

        sf = 100.
        times = np.arange(data.size) / sf

                    #load hypnogram for this same subject
        hypnogram = []
        hypnogram = np.loadtxt(fname = HYPNO[f],dtype = 'str',delimiter = ',')  
        hypnogram = hypnogram.astype('U3')  


        for r in range(len(hypnogram)):
            if hypnogram[r] == '- 1':
                hypnogram[r] = '-1'
            elif hypnogram[r] == '2 1':
                hypnogram[r] = '2'
            elif hypnogram[r] == '-':
                hypnogram[r] = '-1'


        ix = 0
        frame = 30
        itwin = int(frame*sf)

        where = np.isin(hypnogram, ['0','1','2','3','4'])  # True if sample is in N2 / N3, False otherwise
        frontdata = np.average(data[0:2,:],axis=0)
        DSlen = np.ceil(len(frontdata)/itwin)

        fdeltaTS = np.zeros(int(DSlen))

        for wind in range(0,len(frontdata),itwin):

            tempdata = frontdata[wind:wind+itwin]
            tempwhere = where[wind:wind+itwin]

            data_sleep = tempdata[tempwhere]  

            if len(data_sleep) == 0:

                fdeltaTS[ix] = 'NaN'
                ix = ix+1

            else:

                win = int(4 * sf)  # Window size is set to 4 seconds
                freqs, psd = welch(data_sleep, sf, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                fdelta = np.average(psd[4:17])

                fdeltaTS[ix] = fdelta

                ix = ix+1
        
        hypnew = hypnogram.astype('float')
        hypnoDownSamp = np.interp(np.arange(0, len(hypnew), itwin), np.arange(0, len(hypnew)), hypnew)   #downsample hypnogram
        
        
        ## concatenate delta power values from NREM (N2 and N3) stages and out them into a single vector
        ix_d = 0
        NREM_delta = np.zeros(len(fdeltaTS))
        
        for it in range(0,len(hypnoDownSamp)):
            
            if hypnoDownSamp[it] == 2.0:
                
                NREM_delta[ix_d] = fdeltaTS[it]
                ix_d = ix_d + 1
                
            elif hypnoDownSamp[it] == 3.0: 
                
                NREM_delta[ix_d] = fdeltaTS[it]
                ix_d = ix_d + 1
            
        NREM_delta_temp = NREM_delta[NREM_delta != 0]
        NREM_delDeg = np.zeros(len(NREM_delta_temp))
        
        
        
        #compute average and stdev so to remove high power delta bursts near labeled artifacts
        avg_delt = np.average(NREM_delta_temp)
        std_delt = np.std(NREM_delta_temp)
        
        ix_delt = 0
        for de in range(0,len(NREM_delDeg)):
        
            if NREM_delta_temp[de] < (avg_delt + 4*std_delt):
                NREM_delDeg[ix_delt] = NREM_delta_temp[de]
                ix_delt = ix_delt+1
                
        NREM_deltaDegradation = NREM_delDeg[NREM_delDeg != 0]
        
        ## compute the slope of delta change (linear) across the NREM stages
        xx = np.zeros(len(NREM_deltaDegradation),dtype = int)
        for x in range(len(NREM_deltaDegradation)):
            xx[x] = int(x)
            
        slope, intercept, r_value, p_value, std_err = linregress([xx],[NREM_deltaDegradation])
        
        Stats[f+1,0] = Hyp_fnum
        Stats[f+1,1] = slope
        
        
outfile = open('/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/' + 'NREM_DeltaDegradation.csv','w')
with outfile:
    writer = csv.writer(outfile,delimiter=',')
    writer.writerows(Stats)
     
        
        
# plot a single subject's delta power underneath their hypnogram


times = np.zeros(len(fdeltaTS))

ixt = 0
for t in range(0,len(fdeltaTS)):
    times[ixt] = t/(2*60)
    ixt = ixt+1

fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 6),
                                       gridspec_kw={'height_ratios': [1, 2]})
plt.subplots_adjust(hspace=0.1)

# Hypnogram (top axis)
ax0.plot(times,-1 * hypnoDownSamp, color='k')
ax0.set_yticks([1, 0, -1, -2, -3, -4])
ax0.set_yticklabels(['Art', 'W', 'N1', 'N2', 'N3', 'R'])
ax0.set_ylim(-4.5, 1.5)

# Spectrogram (bottom axis)
ax1.plot(times,fdeltaTS[:])
ax1.set_ylabel('Delta Power (µV^2)')
ax1.set_xlabel('Time [hrs]')