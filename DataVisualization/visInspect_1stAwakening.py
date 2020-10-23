# This script is for visually inspecting the PSG data from the 1st awakening of the night
# It will load in each file individually and plot the 1st minute of data from each electrode



from scipy.stats import zscore
from scipy.signal import welch
from scipy.stats import linregress
import scipy.signal as signal
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

def hypno_correction(hypnogram):

    out = []
    hypnogram = hypnogram.astype('U3')
    
    for r in range(len(hypnogram)):
        if hypnogram[r] == '- 1':
            hypnogram[r] = '-1'
        elif hypnogram[r] == '2 1':
            hypnogram[r] = '2'
        elif hypnogram[r] == '-':
            hypnogram[r] = '-1'

    out = hypnogram
    return out
            
            
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
            hypnogram = hypno_correction(hypnogram)

            where_sleep = np.isin(hypnogram, ['1','2','3','4'])  # True if sample is in any sleep stage, False otherwise
            sleep_ix = np.where(where_sleep)    #get a vector of sleep indeces so that I can split the night into before and after initially falling asleep
            sleep_ix = np.transpose(sleep_ix)
            begin_sleep = sleep_ix[0][0]
            end_sleep = sleep_ix[len(sleep_ix)-1][0]

            during_sleep_hypno = hypnogram[begin_sleep:end_sleep]

            during_sleep_data = data[:,begin_sleep:end_sleep]

            where_during_sleep_wakes = np.isin(during_sleep_hypno, ['0']) 

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
            SleepHealth[f+1,3] = ix_awa
            SleepHealth[f+1,4] = np.mean(wake_len_15sec)
            SleepHealth[f+1,5] = np.max(wake_len_15sec)

            #for computing TF metrics on only first min on WASOs
            wake_len_1min = np.nonzero(wake_len > 6000)   
            wake_len_1min = wake_len_1min[0][:]

            wake_bouts = np.asarray(wake_bouts)

            wake_starts = wake_bouts[wake_len_1min,0]

            #create vector bools that is the length of data, default falses anything where they woke up for >1min will be 'True' for 1st min of WASO
            where_1min_WASOs = np.zeros((1,np.shape(during_sleep_data)[1]),dtype=bool)

            ix = 0
            for wk in range(len(where_1min_WASOs[0])):
                if wk == wake_starts[ix]:
                    if ix == 0:
                        where_1min_WASOs[0][wk:wk+6000] = True
                        ix = ix+1

            where_1min_WASOs = np.squeeze(where_1min_WASOs)
            data_during_sleep = during_sleep_data[:, where_1min_WASOs]
  
            fig, ax = plt.subplots(1, 1, figsize=(16, 4))
            plt.plot(data_during_sleep[0,:])
            plt.title('F3 - %s\n' % (Hyp_fnum))
            plt.ylabel('Amplitude (uV)')
            plt.ylim([-100, 100])
            
            fig, ax = plt.subplots(1, 1, figsize=(16, 4))
            plt.plot(data_during_sleep[1,:])
            plt.title('F4- %s\n' % (Hyp_fnum))
            plt.ylabel('Amplitude (uV)')
            plt.ylim([-100, 100])
            
            fig, ax = plt.subplots(1, 1, figsize=(16, 4))
            plt.plot(data_during_sleep[2,:])
            plt.title('C3 - %s\n' % (Hyp_fnum))
            plt.ylabel('Amplitude (uV)')
            plt.ylim([-100, 100])
            
            fig, ax = plt.subplots(1, 1, figsize=(16, 4))
            plt.plot(data_during_sleep[3,:])
            plt.title('C4- %s\n' % (Hyp_fnum))
            plt.ylabel('Amplitude (uV)')
            plt.ylim([-100, 100])
            
            fig, ax = plt.subplots(1, 1, figsize=(16, 4))
            plt.plot(data_during_sleep[4,:])
            plt.title('O1 - %s\n' % (Hyp_fnum))
            plt.ylabel('Amplitude (uV)')
            plt.ylim([-100, 100])
            
            fig, ax = plt.subplots(1, 1, figsize=(16, 4))
            plt.plot(data_during_sleep[5,:])
            plt.title('O2 - %s\n' % (Hyp_fnum))
            plt.ylabel('Amplitude (uV)')
            plt.ylim([-100, 100])