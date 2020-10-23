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



#format output header  -- spectral power for NREM stages
preSleepPow = np.zeros((len(FILES)+1,64), dtype=np.int)
preSleepPow = preSleepPow.astype('U20')

preSleepPow[0,0] = 'Subject'
preSleepPow[0,1] = 'F3 low freq SO'
preSleepPow[0,2] = 'F3 Delta'
preSleepPow[0,3] = 'F3 theta'
preSleepPow[0,4] = 'F3 alpha'
preSleepPow[0,5] = 'F3 sigma'
preSleepPow[0,6] = 'F3 beta'
preSleepPow[0,7] = 'F3 gamma'
preSleepPow[0,8] = 'F4 low freq SO'
preSleepPow[0,9] = 'F4 Delta'
preSleepPow[0,10] = 'F4 theta'
preSleepPow[0,11] = 'F4 alpha'
preSleepPow[0,12] = 'F4 sigma'
preSleepPow[0,13] = 'F4 beta'
preSleepPow[0,14] = 'F4 gamma'
preSleepPow[0,15] = 'C3 low freq SO'
preSleepPow[0,16] = 'C3 Delta'
preSleepPow[0,17] = 'C3 theta'
preSleepPow[0,18] = 'C3 alpha'
preSleepPow[0,19] = 'C3 sigma'
preSleepPow[0,20] = 'C3 beta'
preSleepPow[0,21] = 'C3 gamma'
preSleepPow[0,22] = 'C4 low freq SO'
preSleepPow[0,23] = 'C4 Delta'
preSleepPow[0,24] = 'C4 theta'
preSleepPow[0,25] = 'C4 alpha'
preSleepPow[0,26] = 'C4 sigma'
preSleepPow[0,27] = 'C4 beta'
preSleepPow[0,28] = 'C4 gamma'
preSleepPow[0,29] = 'O1 low freq SO'
preSleepPow[0,30] = 'O1 Delta'
preSleepPow[0,31] = 'O1 theta'
preSleepPow[0,32] = 'O1 alpha'
preSleepPow[0,33] = 'O1 sigma'
preSleepPow[0,34] = 'O1 beta'
preSleepPow[0,35] = 'O1 gamma'
preSleepPow[0,36] = 'O2 low freq SO'
preSleepPow[0,37] = 'O2 Delta'
preSleepPow[0,38] = 'O2 theta'
preSleepPow[0,39] = 'O2 alpha'
preSleepPow[0,40] = 'O2 sigma'
preSleepPow[0,41] = 'O2 beta'
preSleepPow[0,42] = 'O2 gamma'
preSleepPow[0,43] = 'Frontal low freq SO'
preSleepPow[0,44] = 'Frontal Delta'
preSleepPow[0,45] = 'Frontal theta'
preSleepPow[0,46] = 'Frontal alpha'
preSleepPow[0,47] = 'Frontal sigma'
preSleepPow[0,48] = 'Frontal beta'
preSleepPow[0,49] = 'Frontal gamma'
preSleepPow[0,50] = 'Frontal low freq SO'
preSleepPow[0,51] = 'Central Delta'
preSleepPow[0,52] = 'Central theta'
preSleepPow[0,53] = 'Central alpha'
preSleepPow[0,54] = 'Central sigma'
preSleepPow[0,55] = 'Central beta'
preSleepPow[0,56] = 'Central gamma'
preSleepPow[0,57] = 'Posterior low freq SO'
preSleepPow[0,58] = 'Posterior Delta'
preSleepPow[0,59] = 'Posterior theta'
preSleepPow[0,60] = 'Posterior alpha'
preSleepPow[0,61] = 'Posterior sigma'
preSleepPow[0,62] = 'Posterior beta'
preSleepPow[0,63] = 'Posterior gamma'





#format output header  -- spectral power for NREM stages
duringSleepPow = np.zeros((len(FILES)+1,43), dtype=np.int)
duringSleepPow = duringSleepPow.astype('U20')

duringSleepPow[0,0] = 'Subject'
duringSleepPow[0,1] = 'F3 low freq SO'
duringSleepPow[0,2] = 'F3 Delta'
duringSleepPow[0,3] = 'F3 theta'
duringSleepPow[0,4] = 'F3 alpha'
duringSleepPow[0,5] = 'F3 sigma'
duringSleepPow[0,6] = 'F3 beta'
duringSleepPow[0,7] = 'F3 gamma'
duringSleepPow[0,8] = 'F4 low freq SO'
duringSleepPow[0,9] = 'F4 Delta'
duringSleepPow[0,10] = 'F4 theta'
duringSleepPow[0,11] = 'F4 alpha'
duringSleepPow[0,12] = 'F4 sigma'
duringSleepPow[0,13] = 'F4 beta'
duringSleepPow[0,14] = 'F4 gamma'
duringSleepPow[0,15] = 'C3 low freq SO'
duringSleepPow[0,16] = 'C3 Delta'
duringSleepPow[0,17] = 'C3 theta'
duringSleepPow[0,18] = 'C3 alpha'
duringSleepPow[0,19] = 'C3 sigma'
duringSleepPow[0,20] = 'C3 beta'
duringSleepPow[0,21] = 'C3 gamma'
duringSleepPow[0,22] = 'C4 low freq SO'
duringSleepPow[0,23] = 'C4 Delta'
duringSleepPow[0,24] = 'C4 theta'
duringSleepPow[0,25] = 'C4 alpha'
duringSleepPow[0,26] = 'C4 sigma'
duringSleepPow[0,27] = 'C4 beta'
duringSleepPow[0,28] = 'C4 gamma'
duringSleepPow[0,29] = 'O1 low freq SO'
duringSleepPow[0,30] = 'O1 Delta'
duringSleepPow[0,31] = 'O1 theta'
duringSleepPow[0,32] = 'O1 alpha'
duringSleepPow[0,33] = 'O1 sigma'
duringSleepPow[0,34] = 'O1 beta'
duringSleepPow[0,35] = 'O1 gamma'
duringSleepPow[0,36] = 'O2 low freq SO'
duringSleepPow[0,37] = 'O2 Delta'
duringSleepPow[0,38] = 'O2 theta'
duringSleepPow[0,39] = 'O2 alpha'
duringSleepPow[0,40] = 'O2 sigma'
duringSleepPow[0,41] = 'O2 beta'
duringSleepPow[0,42] = 'O2 gamma'

#format output header  -- spectral power for NREM stages
SleepHealth = np.zeros((len(FILES)+1,10), dtype=np.int)
SleepHealth = SleepHealth.astype('U20')

SleepHealth[0,0] = 'Subject'
SleepHealth[0,1] = 'Sleep Latency'
SleepHealth[0,2] = 'Total WASO min'
SleepHealth[0,3] = 'Total num WASO instances'
SleepHealth[0,4] = 'Avg len WASO'
SleepHealth[0,5] = 'Max len WASO'
SleepHealth[0,6] = 'Total num WASO instances 15 sec cutoff'
SleepHealth[0,7] = 'Avg len WASO 15 sec cutoff'
SleepHealth[0,8] = 'Max len WASO 15 sec cutoff'
SleepHealth[0,9] = 'Theta 1 min WASO slope'

#find continuous regions of WASO (start and stop indexes) in sleep data
def contiguous_regions(condition):
    idx = []
    i = 0
    while i < len(condition):
        x1 = i + condition[i:].argmax()
        try:
            x2 = x1 + condition[x1:].argmin()
        except:
            x2 = x1 + 1
        if x1 == x2:
            if condition[x1] == True:
                x2 = len(condition)
            else:
                break
        idx.append( [x1,x2] )
        i = x2
    return idx


#create a variable that will contain a theta TS (all elecs averaged) for each sub
GlobalTheta = np.zeros((len(FILES),30),dtype=float)

## initialize for loop
for f, file in enumerate(FILES):

    if (f != 41):
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
                    
            
            where_sleep = np.isin(hypnogram, ['1','2','3','4'])  # True if sample is in any sleep stage, False otherwise
            sleep_ix = np.where(where_sleep)    #get a vector of sleep indeces so that I can split the night into before and after initially falling asleep
            sleep_ix = np.transpose(sleep_ix)
            begin_sleep = sleep_ix[0][0]
            end_sleep = sleep_ix[len(sleep_ix)-1][0]
            
            before_sleep_hypno = hypnogram[begin_sleep-6000:begin_sleep]
            during_sleep_hypno = hypnogram[begin_sleep:end_sleep]
            
            before_sleep_data = data[:,begin_sleep-6000:begin_sleep]
            during_sleep_data = data[:,begin_sleep:end_sleep]
            
           # where_pre_sleep_wakes = np.isin(before_sleep_hypno, ['0'])  
            where_during_sleep_wakes = np.isin(during_sleep_hypno, ['0']) 
            
          #  data_pre_sleep = before_sleep_data[:, where_pre_sleep_wakes]
            data_during_sleep = during_sleep_data[:, where_during_sleep_wakes]
        
            wake_bouts = contiguous_regions(where_during_sleep_wakes)

            wake_len = np.zeros((len(wake_bouts),1), dtype=np.float)
            
            ix_awa = 0
            for w in range(len(wake_bouts)):
                wake_len[w] = (wake_bouts[w][1] - wake_bouts[w][0])   #WASO lengths in min
                
                if wake_len[w] > 1500:   #if greater than 15 sec, count as awakening
                    ix_awa = ix_awa + 1
                
            #for removing wakes less than 3 sec long
            wake_len_15sec = np.nonzero(wake_len > 1500)   #index WASOs greater than 3 sec -- Sleep Fragmentation, Mezick 2013
            wake_len_15sec = wake_len_15sec[0][:]
            wake_len_15sec = wake_len[wake_len_15sec]
            
            
            SleepHealth[f+1,0] = Hyp_fnum
            SleepHealth[f+1,1] = begin_sleep/(sf*60)
            SleepHealth[f+1,2] = len(data_during_sleep[1])/(60*sf)
            SleepHealth[f+1,3] = len(wake_bouts)
            SleepHealth[f+1,4] = np.mean(wake_len)
            SleepHealth[f+1,5] = np.max(wake_len)
            
            SleepHealth[f+1,6] = ix_awa
            SleepHealth[f+1,7] = np.mean(wake_len_15sec)
            SleepHealth[f+1,8] = np.max(wake_len_15sec)
            
            
            #for computing TF metrics on only first min on WASOs
            wake_len_1min = np.nonzero(wake_len > 6000)   
            wake_len_1min = wake_len_1min[0][:]
            
            wake_bouts = np.asarray(wake_bouts)
            
            wake_starts = wake_bouts[wake_len_1min,0]
            
            #create vector bools that is the length of data, default falses anything where they woke up for >1min will be 'True' for 1st min of WASO
            where_1min_WASOs = np.zeros((1,np.shape(during_sleep_data)[1]),dtype=bool)
            
            ix = 0
            for wk in range(len(where_1min_WASOs[0])):
                if wk == wake_starts[ix-1]:
                    where_1min_WASOs[0][wk:wk+6000] = True
                    ix = ix+1
                    
            where_1min_WASOs = np.squeeze(where_1min_WASOs)
            data_during_sleep = during_sleep_data[:, where_1min_WASOs]
            
            ## this will compute TF data on the first min of WASOs in 2 sec time bins, save the time series of theta for each person, and compute theta slope during this first min of WASO
            
            
            frame = 2
            itwin = int(frame*sf)

            DSlen = np.ceil(len(data_during_sleep[1])/itwin)

            thetaTS = np.zeros((6,int(DSlen)),dtype=float)

            for elec in range(data_during_sleep.shape[0]):
                ix = 0
                elec_thetaTS = np.zeros(int(DSlen),dtype=float)
                
                for wind in range(0,data_during_sleep.shape[1],itwin):

                    tempdata = data_during_sleep[elec,wind:wind+itwin]

                    win = int(2 * sf)  # Window size is set to 2 seconds
                    freqs, psd = welch(tempdata, sf, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    theta = np.average(psd[8:15])

                    thetaTS[elec,ix] = theta    #this one will store an electrode x time matrix that all be saved independently for each subject. This way, we have either a subject x electrode x time matrix, or keep the elec x time matrix but make 6 rows for each subject
                    
                    elec_thetaTS[ix] = theta   #this one is for computing slope

                    ix = ix+1
                    
            if thetaTS[1].shape[0] > 0:
                
                GlobalTheta[f,:] = thetaTS.mean(axis=0)

                ## compute the slope of delta change (linear) across the NREM stages
                xx = np.zeros(GlobalTheta[f,:].shape[0],dtype = float)
                for x in range(GlobalTheta[f,:].shape[0]):
                    xx[x] = int(x)

                slope, intercept, r_value, p_value, std_err = linregress([xx],[GlobalTheta[f,:]])

                SleepHealth[f+1,9] = slope

                ## plot these and run stats -- then I'm done!
                

                
                
            
            ### this does normal TF analysis on WASO data
#             if len(data_during_sleep[1]) > 1:
#                 duringsleep_freqs, duringsleep_psd = welch(data_during_sleep, sf, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

#                 duringsleep_df = yasa.bandpower_from_psd(duringsleep_psd, duringsleep_freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                
#                 duringSleepPow[f+1,0] = Hyp_fnum

#                     #F3
#                 duringSleepPow[f+1,1] = duringsleep_df.iloc[0,1]
#                 duringSleepPow[f+1,2] = duringsleep_df.iloc[0,2]
#                 duringSleepPow[f+1,3] = duringsleep_df.iloc[0,3]
#                 duringSleepPow[f+1,4] = duringsleep_df.iloc[0,4]
#                 duringSleepPow[f+1,5] = duringsleep_df.iloc[0,5]
#                 duringSleepPow[f+1,6] = duringsleep_df.iloc[0,6]
#                 duringSleepPow[f+1,7] = duringsleep_df.iloc[0,7]

#                    # F4
#                 duringSleepPow[f+1,8] = duringsleep_df.iloc[1,1]
#                 duringSleepPow[f+1,9] = duringsleep_df.iloc[1,2]
#                 duringSleepPow[f+1,10] = duringsleep_df.iloc[1,3]
#                 duringSleepPow[f+1,11] = duringsleep_df.iloc[1,4]
#                 duringSleepPow[f+1,12] = duringsleep_df.iloc[1,5]
#                 duringSleepPow[f+1,13] = duringsleep_df.iloc[1,6]
#                 duringSleepPow[f+1,14] = duringsleep_df.iloc[1,7]

#                     #C3
#                 duringSleepPow[f+1,15] = duringsleep_df.iloc[2,1]
#                 duringSleepPow[f+1,16] = duringsleep_df.iloc[2,2]
#                 duringSleepPow[f+1,17] = duringsleep_df.iloc[2,3]
#                 duringSleepPow[f+1,18] = duringsleep_df.iloc[2,4]
#                 duringSleepPow[f+1,19] = duringsleep_df.iloc[2,5]
#                 duringSleepPow[f+1,20] = duringsleep_df.iloc[2,6]
#                 duringSleepPow[f+1,21] = duringsleep_df.iloc[2,7]

#                    # C4
#                 duringSleepPow[f+1,22] = duringsleep_df.iloc[3,1]
#                 duringSleepPow[f+1,23] = duringsleep_df.iloc[3,2]
#                 duringSleepPow[f+1,24] = duringsleep_df.iloc[3,3]
#                 duringSleepPow[f+1,25] = duringsleep_df.iloc[3,4]
#                 duringSleepPow[f+1,26] = duringsleep_df.iloc[3,5]
#                 duringSleepPow[f+1,27] = duringsleep_df.iloc[3,6]
#                 duringSleepPow[f+1,28] = duringsleep_df.iloc[3,7]

#                    # O1
#                 duringSleepPow[f+1,29] = duringsleep_df.iloc[4,1]
#                 duringSleepPow[f+1,30] = duringsleep_df.iloc[4,2]
#                 duringSleepPow[f+1,31] = duringsleep_df.iloc[4,3]
#                 duringSleepPow[f+1,32] = duringsleep_df.iloc[4,4]
#                 duringSleepPow[f+1,33] = duringsleep_df.iloc[4,5]
#                 duringSleepPow[f+1,34] = duringsleep_df.iloc[4,6]
#                 duringSleepPow[f+1,35] = duringsleep_df.iloc[4,7]

#                    # O2
#                 duringSleepPow[f+1,36] = duringsleep_df.iloc[5,1]
#                 duringSleepPow[f+1,37] = duringsleep_df.iloc[5,2]
#                 duringSleepPow[f+1,38] = duringsleep_df.iloc[5,3]
#                 duringSleepPow[f+1,39] = duringsleep_df.iloc[5,4]
#                 duringSleepPow[f+1,40] = duringsleep_df.iloc[5,5]
#                 duringSleepPow[f+1,41] = duringsleep_df.iloc[5,6]
#                 duringSleepPow[f+1,42] = duringsleep_df.iloc[5,7]
            
            
#             if len(data_pre_sleep[1]) > 0:   #did they go to sleep at all?
#                 win = int(4 * sf)  # Window size is set to 4 seconds
#                 presleep_freqs, presleep_psd = welch(data_pre_sleep, sf, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

#                 presleep_df = yasa.bandpower_from_psd(presleep_psd, presleep_freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)

#                # duringsleep_freqs, duringsleep_psd = welch(data_during_sleep, sf, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

#               #  duringsleep_df = yasa.bandpower_from_psd(duringsleep_psd, duringsleep_freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)


#                     #pull power stats from stage 2 and 3 and store them
#                 preSleepPow[f+1,0] = Hyp_fnum
                
#                     #F3
#                 preSleepPow[f+1,1] = presleep_df.iloc[0,1]
#                 preSleepPow[f+1,2] = presleep_df.iloc[0,2]
#                 preSleepPow[f+1,3] = presleep_df.iloc[0,3]
#                 preSleepPow[f+1,4] = presleep_df.iloc[0,4]
#                 preSleepPow[f+1,5] = presleep_df.iloc[0,5]
#                 preSleepPow[f+1,6] = presleep_df.iloc[0,6]
#                 preSleepPow[f+1,7] = presleep_df.iloc[0,7]

#                     #F4
#                 preSleepPow[f+1,8] = presleep_df.iloc[1,1]
#                 preSleepPow[f+1,9] = presleep_df.iloc[1,2]
#                 preSleepPow[f+1,10] = presleep_df.iloc[1,3]
#                 preSleepPow[f+1,11] = presleep_df.iloc[1,4]
#                 preSleepPow[f+1,12] = presleep_df.iloc[1,5]
#                 preSleepPow[f+1,13] = presleep_df.iloc[1,6]
#                 preSleepPow[f+1,14] = presleep_df.iloc[1,7]

#                     #C3
#                 preSleepPow[f+1,15] = presleep_df.iloc[2,1]
#                 preSleepPow[f+1,16] = presleep_df.iloc[2,2]
#                 preSleepPow[f+1,17] = presleep_df.iloc[2,3]
#                 preSleepPow[f+1,18] = presleep_df.iloc[2,4]
#                 preSleepPow[f+1,19] = presleep_df.iloc[2,5]
#                 preSleepPow[f+1,20] = presleep_df.iloc[2,6]
#                 preSleepPow[f+1,21] = presleep_df.iloc[2,7]

#                     #C4
#                 preSleepPow[f+1,22] = presleep_df.iloc[3,1]
#                 preSleepPow[f+1,23] = presleep_df.iloc[3,2]
#                 preSleepPow[f+1,24] = presleep_df.iloc[3,3]
#                 preSleepPow[f+1,25] = presleep_df.iloc[3,4]
#                 preSleepPow[f+1,26] = presleep_df.iloc[3,5]
#                 preSleepPow[f+1,27] = presleep_df.iloc[3,6]
#                 preSleepPow[f+1,28] = presleep_df.iloc[3,7]

#                     #O1
#                 preSleepPow[f+1,29] = presleep_df.iloc[4,1]
#                 preSleepPow[f+1,30] = presleep_df.iloc[4,2]
#                 preSleepPow[f+1,31] = presleep_df.iloc[4,3]
#                 preSleepPow[f+1,32] = presleep_df.iloc[4,4]
#                 preSleepPow[f+1,33] = presleep_df.iloc[4,5]
#                 preSleepPow[f+1,34] = presleep_df.iloc[4,6]
#                 preSleepPow[f+1,35] = presleep_df.iloc[4,7]

#                     #O2
#                 preSleepPow[f+1,36] = presleep_df.iloc[5,1]
#                 preSleepPow[f+1,37] = presleep_df.iloc[5,2]
#                 preSleepPow[f+1,38] = presleep_df.iloc[5,3]
#                 preSleepPow[f+1,39] = presleep_df.iloc[5,4]
#                 preSleepPow[f+1,40] = presleep_df.iloc[5,5]
#                 preSleepPow[f+1,41] = presleep_df.iloc[5,6]
#                 preSleepPow[f+1,42] = presleep_df.iloc[5,7]
                
#                 try:
#                     #Frontal
#                     preSleepPow[f+1,43] = (float(preSleepPow[f+1,1]) + float(preSleepPow[f+1,8]))/2
#                     preSleepPow[f+1,44] = (float(preSleepPow[f+1,2]) + float(preSleepPow[f+1,9]))/2
#                     preSleepPow[f+1,45] = (float(preSleepPow[f+1,3]) + float(preSleepPow[f+1,10]))/2
#                     preSleepPow[f+1,46] = (float(preSleepPow[f+1,4]) + float(preSleepPow[f+1,11]))/2
#                     preSleepPow[f+1,47] = (float(preSleepPow[f+1,5]) + float(preSleepPow[f+1,12]))/2
#                     preSleepPow[f+1,48] = (float(preSleepPow[f+1,6]) + float(preSleepPow[f+1,13]))/2
#                     preSleepPow[f+1,49] = (float(preSleepPow[f+1,7]) + float(preSleepPow[f+1,14]))/2

#                     #Central
#                     preSleepPow[f+1,50] = (float(preSleepPow[f+1,15]) + float(preSleepPow[f+1,22]))/2
#                     preSleepPow[f+1,51] = (float(preSleepPow[f+1,16]) + float(preSleepPow[f+1,23]))/2
#                     preSleepPow[f+1,52] = (float(preSleepPow[f+1,17]) + float(preSleepPow[f+1,24]))/2
#                     preSleepPow[f+1,53] = (float(preSleepPow[f+1,18]) + float(preSleepPow[f+1,25]))/2
#                     preSleepPow[f+1,54] = (float(preSleepPow[f+1,19]) + float(preSleepPow[f+1,26]))/2
#                     preSleepPow[f+1,55] = (float(preSleepPow[f+1,20]) + float(preSleepPow[f+1,27]))/2
#                     preSleepPow[f+1,56] = (float(preSleepPow[f+1,21]) + float(preSleepPow[f+1,28]))/2

#                     #Posterior
#                     preSleepPow[f+1,57] = (float(preSleepPow[f+1,29]) + float(preSleepPow[f+1,36]))/2
#                     preSleepPow[f+1,58] = (float(preSleepPow[f+1,30]) + float(preSleepPow[f+1,37]))/2
#                     preSleepPow[f+1,59] = (float(preSleepPow[f+1,31]) + float(preSleepPow[f+1,38]))/2
#                     preSleepPow[f+1,60] = (float(preSleepPow[f+1,32]) + float(preSleepPow[f+1,39]))/2
#                     preSleepPow[f+1,61] = (float(preSleepPow[f+1,33]) + float(preSleepPow[f+1,40]))/2
#                     preSleepPow[f+1,62] = (float(preSleepPow[f+1,34]) + float(preSleepPow[f+1,41]))/2
#                     preSleepPow[f+1,63] = (float(preSleepPow[f+1,35]) + float(preSleepPow[f+1,42]))/2



                    #pull power stats from stage 2 and 3 and store them
            #    duringSleepPow[f+1,0] = Hyp_fnum

                    #F3
            #    duringSleepPow[f+1,1] = duringsleep_df.iloc[0,1]
            #    duringSleepPow[f+1,2] = duringsleep_df.iloc[0,2]
            #    duringSleepPow[f+1,3] = duringsleep_df.iloc[0,3]
            #    duringSleepPow[f+1,4] = duringsleep_df.iloc[0,4]
            #    duringSleepPow[f+1,5] = duringsleep_df.iloc[0,5]
            #    duringSleepPow[f+1,6] = duringsleep_df.iloc[0,6]
            #    duringSleepPow[f+1,7] = duringsleep_df.iloc[0,7]

                    #F4
            #    duringSleepPow[f+1,8] = duringsleep_df.iloc[1,1]
            #    duringSleepPow[f+1,9] = duringsleep_df.iloc[1,2]
            #    duringSleepPow[f+1,10] = duringsleep_df.iloc[1,3]
            #    duringSleepPow[f+1,11] = duringsleep_df.iloc[1,4]
            #    duringSleepPow[f+1,12] = duringsleep_df.iloc[1,5]
            #    duringSleepPow[f+1,13] = duringsleep_df.iloc[1,6]
            #    duringSleepPow[f+1,14] = duringsleep_df.iloc[1,7]

                    #C3
            #    duringSleepPow[f+1,15] = duringsleep_df.iloc[2,1]
            #    duringSleepPow[f+1,16] = duringsleep_df.iloc[2,2]
            #    duringSleepPow[f+1,17] = duringsleep_df.iloc[2,3]
            #    duringSleepPow[f+1,18] = duringsleep_df.iloc[2,4]
            #    duringSleepPow[f+1,19] = duringsleep_df.iloc[2,5]
            #    duringSleepPow[f+1,20] = duringsleep_df.iloc[2,6]
            #    duringSleepPow[f+1,21] = duringsleep_df.iloc[2,7]

                    #C4
            #    duringSleepPow[f+1,22] = duringsleep_df.iloc[3,1]
            #    duringSleepPow[f+1,23] = duringsleep_df.iloc[3,2]
            #    duringSleepPow[f+1,24] = duringsleep_df.iloc[3,3]
            #    duringSleepPow[f+1,25] = duringsleep_df.iloc[3,4]
            #    duringSleepPow[f+1,26] = duringsleep_df.iloc[3,5]
            #    duringSleepPow[f+1,27] = duringsleep_df.iloc[3,6]
            #    duringSleepPow[f+1,28] = duringsleep_df.iloc[3,7]

                    #O1
            #    duringSleepPow[f+1,29] = duringsleep_df.iloc[4,1]
            #    duringSleepPow[f+1,30] = duringsleep_df.iloc[4,2]
            #    duringSleepPow[f+1,31] = duringsleep_df.iloc[4,3]
            #    duringSleepPow[f+1,32] = duringsleep_df.iloc[4,4]
            #    duringSleepPow[f+1,33] = duringsleep_df.iloc[4,5]
            #    duringSleepPow[f+1,34] = duringsleep_df.iloc[4,6]
            #    duringSleepPow[f+1,35] = duringsleep_df.iloc[4,7]

                    #O2
            #    duringSleepPow[f+1,36] = duringsleep_df.iloc[5,1]
            #    duringSleepPow[f+1,37] = duringsleep_df.iloc[5,2]
            #    duringSleepPow[f+1,38] = duringsleep_df.iloc[5,3]
            #    duringSleepPow[f+1,39] = duringsleep_df.iloc[5,4]
            #    duringSleepPow[f+1,40] = duringsleep_df.iloc[5,5]
            #    duringSleepPow[f+1,41] = duringsleep_df.iloc[5,6]
            #    duringSleepPow[f+1,42] = duringsleep_df.iloc[5,7]
            
            
            
pre_outfile = open('/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/' + 'WASO stats 15 sec threshold.csv','w')
with pre_outfile:
    writer = csv.writer(pre_outfile,delimiter=',')
    writer.writerows(SleepHealth)
    
    
during_outfile = open('/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/' + 'TF 1 min WASO theta time series.csv','w')
with during_outfile:
    writer = csv.writer(during_outfile,delimiter=',')
    writer.writerows(GlobalTheta)