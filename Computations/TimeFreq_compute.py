# this computes time frequency power from PSG data. The stage(s) that you want to compute, must be specified by the user in the variable where_REM
# if it breaks, it's because that subject didn't go into the stage we're interested in. Specify to skip them on line 205 (e.g., if (f != 106))

from scipy.stats import zscore
from scipy.signal import welch
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

PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/Preprocessed/'

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
SWstats = np.zeros((len(FILES)+1,9), dtype=np.int)
SWstats = SWstats.astype('U20')

SWstats[0,0] = 'Subject'
SWstats[0,1] = 'S2 C3 SW density'
SWstats[0,2] = 'S2 C4 SW density'
SWstats[0,3] = 'S2 F3 SW density'
SWstats[0,4] = 'S2 F4 SW density'
SWstats[0,5] = 'S3 C3 SW density'
SWstats[0,6] = 'S3 C4 SW density'
SWstats[0,7] = 'S3 F3 SW density'
SWstats[0,8] = 'S3 F4 SW density'

#format output header  -- spectral power for NREM stages
PowStats = np.zeros((len(FILES)+1,43), dtype=np.int)
PowStats = PowStats.astype('U20')

PowStats[0,0] = 'Subject'
PowStats[0,1] = 'F3 low freq SO'
PowStats[0,2] = 'F3 Delta'
PowStats[0,3] = 'F3 theta'
PowStats[0,4] = 'F3 alpha'
PowStats[0,5] = 'F3 sigma'
PowStats[0,6] = 'F3 beta'
PowStats[0,7] = 'F3 gamma'
PowStats[0,8] = 'F4 low freq SO'
PowStats[0,9] = 'F4 Delta'
PowStats[0,10] = 'F4 theta'
PowStats[0,11] = 'F4 alpha'
PowStats[0,12] = 'F4 sigma'
PowStats[0,13] = 'F4 beta'
PowStats[0,14] = 'F4 gamma'
PowStats[0,15] = 'C3 low freq SO'
PowStats[0,16] = 'C3 Delta'
PowStats[0,17] = 'C3 theta'
PowStats[0,18] = 'C3 alpha'
PowStats[0,19] = 'C3 sigma'
PowStats[0,20] = 'C3 beta'
PowStats[0,21] = 'C3 gamma'
PowStats[0,22] = 'C4 low freq SO'
PowStats[0,23] = 'C4 Delta'
PowStats[0,24] = 'C4 theta'
PowStats[0,25] = 'C4 alpha'
PowStats[0,26] = 'C4 sigma'
PowStats[0,27] = 'C4 beta'
PowStats[0,28] = 'C4 gamma'
PowStats[0,29] = 'O1 low freq SO'
PowStats[0,30] = 'O1 Delta'
PowStats[0,31] = 'O1 theta'
PowStats[0,32] = 'O1 alpha'
PowStats[0,33] = 'O1 sigma'
PowStats[0,34] = 'O1 beta'
PowStats[0,35] = 'O1 gamma'
PowStats[0,36] = 'O2 low freq SO'
PowStats[0,37] = 'O2 Delta'
PowStats[0,38] = 'O2 theta'
PowStats[0,39] = 'O2 alpha'
PowStats[0,40] = 'O2 sigma'
PowStats[0,41] = 'O2 beta'
PowStats[0,42] = 'O2 gamma'


## initialize for loop
for f, file in enumerate(FILES):

    if (f < 10000):    #unindent not working, too lazy to delete this
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
#             hypnogram = hypnogram.astype('U3')

            for r in range(len(hypnogram)):
                if hypnogram[r] == '- 1':
                    hypnogram[r] = '-1'
                elif hypnogram[r] == '2 1':
                    hypnogram[r] = '2'
                elif hypnogram[r] == '-':
                    hypnogram[r] = '-1'
                elif hypnogram[r] == '0 . 0':
                    hypnogram[r] = '0'

            frontocents = channels[0:4]

#                 # Z-score the data
#             data_zscored = zscore(data)

#             sw = yasa.sw_detect(data_zscored[0:4,:], sf, ch_names=frontocents, hypno=hypnogram, include=(2,3),
#                            amp_neg=(1, None), 
#                            amp_pos=(1, None), 
#                            amp_ptp=(3, 10))

#                 #sw.summary().round(2)

#             summary_df = sw.summary(grp_chan=True, grp_stage=True, aggfunc='mean')

#                 #pull SW density from stage 2 and 3 and store them
#             SWstats[f+1,0] = Hyp_fnum
#             SWstats[f+1,1] = summary_df.iloc[0,1]
#             SWstats[f+1,2] = summary_df.iloc[1,1]
#             SWstats[f+1,3] = summary_df.iloc[2,1]
#             SWstats[f+1,4] = summary_df.iloc[3,1]

#             try:
#                 SWstats[f+1,5] = summary_df.iloc[4,1]
#                 SWstats[f+1,6] = summary_df.iloc[5,1]
#                 SWstats[f+1,7] = summary_df.iloc[6,1]
#                 SWstats[f+1,8] = summary_df.iloc[7,1]
#             except:
#                 SWstats[f+1,5] = 0
#                 SWstats[f+1,6] = 0
#                 SWstats[f+1,7] = 0
#                 SWstats[f+1,8] = 0

            #compute band power
            where_REM = np.isin(hypnogram, ['0'])  # True if sample is in N2 / N3, False otherwise
            data_REM = data[:, where_REM]        

            win = int(4 * sf)  # Window size is set to 4 seconds
            freqs, psd = welch(data_REM, sf, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

            pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)

                #pull power stats from stage 2 and 3 and store them
            PowStats[f+1,0] = Hyp_fnum

                #F3
            PowStats[f+1,1] = pow_df.iloc[0,1]
            PowStats[f+1,2] = pow_df.iloc[0,2]
            PowStats[f+1,3] = pow_df.iloc[0,3]
            PowStats[f+1,4] = pow_df.iloc[0,4]
            PowStats[f+1,5] = pow_df.iloc[0,5]
            PowStats[f+1,6] = pow_df.iloc[0,6]
            PowStats[f+1,7] = pow_df.iloc[0,7]

                #F4
            PowStats[f+1,8] = pow_df.iloc[1,1]
            PowStats[f+1,9] = pow_df.iloc[1,2]
            PowStats[f+1,10] = pow_df.iloc[1,3]
            PowStats[f+1,11] = pow_df.iloc[1,4]
            PowStats[f+1,12] = pow_df.iloc[1,5]
            PowStats[f+1,13] = pow_df.iloc[1,6]
            PowStats[f+1,14] = pow_df.iloc[1,7]

                #C3
            PowStats[f+1,15] = pow_df.iloc[2,1]
            PowStats[f+1,16] = pow_df.iloc[2,2]
            PowStats[f+1,17] = pow_df.iloc[2,3]
            PowStats[f+1,18] = pow_df.iloc[2,4]
            PowStats[f+1,19] = pow_df.iloc[2,5]
            PowStats[f+1,20] = pow_df.iloc[2,6]
            PowStats[f+1,21] = pow_df.iloc[2,7]

                #C4
            PowStats[f+1,22] = pow_df.iloc[3,1]
            PowStats[f+1,23] = pow_df.iloc[3,2]
            PowStats[f+1,24] = pow_df.iloc[3,3]
            PowStats[f+1,25] = pow_df.iloc[3,4]
            PowStats[f+1,26] = pow_df.iloc[3,5]
            PowStats[f+1,27] = pow_df.iloc[3,6]
            PowStats[f+1,28] = pow_df.iloc[3,7]

                #O1
            PowStats[f+1,29] = pow_df.iloc[4,1]
            PowStats[f+1,30] = pow_df.iloc[4,2]
            PowStats[f+1,31] = pow_df.iloc[4,3]
            PowStats[f+1,32] = pow_df.iloc[4,4]
            PowStats[f+1,33] = pow_df.iloc[4,5]
            PowStats[f+1,34] = pow_df.iloc[4,6]
            PowStats[f+1,35] = pow_df.iloc[4,7]

                #O2
            PowStats[f+1,36] = pow_df.iloc[5,1]
            PowStats[f+1,37] = pow_df.iloc[5,2]
            PowStats[f+1,38] = pow_df.iloc[5,3]
            PowStats[f+1,39] = pow_df.iloc[5,4]
            PowStats[f+1,40] = pow_df.iloc[5,5]
            PowStats[f+1,41] = pow_df.iloc[5,6]
            PowStats[f+1,42] = pow_df.iloc[5,7]

        
        
#SW_outfile = open('/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/hol_up/' + 'Slow wave density stats.csv','w')
#with SW_outfile:
#    writer = csv.writer(SW_outfile,delimiter=',')
#    writer.writerows(SWstats)
    
    
SpecPow_outfile = open('/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/Resting/preprocessed/EO_all_1p5_1p25_1p0_stdev/' + 'spectral stats.csv','w')
with SpecPow_outfile:
    writer = csv.writer(SpecPow_outfile,delimiter=',')
    writer.writerows(PowStats)
    
        
        ## Plot 1/f
        #plt.plot(freqs, psd[1], 'k', lw=2)
        #plt.fill_between(freqs, psd[1], cmap='Spectral')
        #plt.xlim(0, 50)
        #plt.yscale('log')
        #sns.despine()
        #plt.title(chan[1])
        #plt.xlabel('Frequency [Hz]')
        #plt.ylabel('PSD log($uV^2$/Hz)');
        
        
        
# Let's get a mask indicating for each sample
#mask = sw.get_mask()
#mask

#sw_highlight = data * mask
#sw_highlight[sw_highlight == 0] = np.nan

#plt.figure(figsize=(16, 4.5))

#plt.plot(times, data, 'k')
#plt.plot(times, sw_highlight, 'indianred')
#plt.plot(events['NegPeak'], sw_highlight[(events['NegPeak'] * sf).astype(int)], 'bo', label='Negative peaks')
#plt.plot(events['PosPeak'], sw_highlight[(events['PosPeak'] * sf).astype(int)], 'go', label='Positive peaks')
#plt.plot(events['Start'], data[(events['Start'] * sf).astype(int)], 'ro', label='Start')

#plt.xlabel('Time (seconds)')
#plt.ylabel('Amplitude (uV)')
#plt.xlim([0, times[-1]])
#plt.title('N3 sleep EEG data')
#plt.legend()
#sns.despine()