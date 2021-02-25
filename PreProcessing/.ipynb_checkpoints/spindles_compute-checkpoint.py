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
spindlestats = np.zeros((len(FILES)+1,41), dtype=np.int)
spindlestats = spindlestats.astype('U20')

spindlestats[0,0] = 'Subject'
spindlestats[0,1] = 'S2 C3 spindle count'
spindlestats[0,2] = 'S2 C3 spindle density'
spindlestats[0,3] = 'S2 C3 spindle duration'
spindlestats[0,4] = 'S2 C3 spindle amplitude'
spindlestats[0,5] = 'S2 C3 spindle freq'

spindlestats[0,6] = 'S2 C4 spindle count'
spindlestats[0,7] = 'S2 C4 spindle density'
spindlestats[0,8] = 'S2 C4 spindle duration'
spindlestats[0,9] = 'S2 C4 spindle amplitude'
spindlestats[0,10] = 'S2 C4 spindle freq'

spindlestats[0,11] = 'S2 F3 spindle count'
spindlestats[0,12] = 'S2 F3 spindle density'
spindlestats[0,13] = 'S2 F3 spindle duration'
spindlestats[0,14] = 'S2 F3 spindle amplitude'
spindlestats[0,15] = 'S2 F3 spindle freq'

spindlestats[0,16] = 'S2 F4 spindle count'
spindlestats[0,17] = 'S2 F4 spindle density'
spindlestats[0,18] = 'S2 F4 spindle duration'
spindlestats[0,19] = 'S2 F4 spindle amplitude'
spindlestats[0,20] = 'S2 F4 spindle freq'

spindlestats[0,21] = 'S3 C3 spindle count'
spindlestats[0,22] = 'S3 C3 spindle density'
spindlestats[0,23] = 'S3 C3 spindle duration'
spindlestats[0,24] = 'S3 C3 spindle amplitude'
spindlestats[0,25] = 'S3 C3 spindle freq'

spindlestats[0,26] = 'S3 C4 spindle density'
spindlestats[0,27] = 'S3 C4 spindle density'
spindlestats[0,28] = 'S3 C4 spindle duration'
spindlestats[0,29] = 'S3 C4 spindle amplitude'
spindlestats[0,30] = 'S3 C4 spindle freq'

spindlestats[0,31] = 'S3 F3 spindle density'
spindlestats[0,32] = 'S3 F3 spindle density'
spindlestats[0,33] = 'S3 F3 spindle duration'
spindlestats[0,34] = 'S3 F3 spindle amplitude'
spindlestats[0,35] = 'S3 F3 spindle freq'

spindlestats[0,36] = 'S3 F4 spindle density'
spindlestats[0,37] = 'S3 F4 spindle density'
spindlestats[0,38] = 'S3 F4 spindle duration'
spindlestats[0,39] = 'S3 F4 spindle amplitude'
spindlestats[0,40] = 'S3 F4 spindle freq'


## initialize for loop
for f, file in enumerate(FILES):

    if (f > 46):    #unindent not working, too lazy to delete this
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
                elif hypnogram[r] == '0 . 0':
                    hypnogram[r] = '0'

            frontocents = channels[0:4]

                # Z-score the data
#             data_zscored = zscore(data)

#             sw = yasa.sw_detect(data_zscored[0:4,:], sf, ch_names=frontocents, hypno=hypnogram, include=(2,3),
#                            amp_neg=(1, None), 
#                            amp_pos=(1, None), 
#                            amp_ptp=(3, 10))

                #sw.summary().round(2)

#             summary_df = sw.summary(grp_chan=True, grp_stage=True, aggfunc='mean')
            
            try:
                sp = yasa.spindles_detect(data, sf, ch_names=channels, hypno=hypnogram, include=(2, 3))

                # Get the full detection dataframe

                spindle_df = sp.summary(grp_chan=True, grp_stage=True, aggfunc='mean')

                    #pull spindle density from stage 2 and 3 and store them
                spindlestats[f+1,0] = Hyp_fnum

                try:
                    spindlestats[f+1,1] = spindle_df.iloc[0,0]
                    spindlestats[f+1,2] = spindle_df.iloc[0,1]
                    spindlestats[f+1,3] = spindle_df.iloc[0,2]
                    spindlestats[f+1,4] = spindle_df.iloc[0,3]
                    spindlestats[f+1,5] = spindle_df.iloc[0,7]
                except:
                    spindlestats[f+1,1] = 0
                    spindlestats[f+1,2] = 0
                    spindlestats[f+1,3] = 0
                    spindlestats[f+1,4] = 0
                    spindlestats[f+1,5] = 0

                try:
                    spindlestats[f+1,6] = spindle_df.iloc[1,0]
                    spindlestats[f+1,7] = spindle_df.iloc[1,1]
                    spindlestats[f+1,8] = spindle_df.iloc[1,2]
                    spindlestats[f+1,9] = spindle_df.iloc[1,3]
                    spindlestats[f+1,10] = spindle_df.iloc[1,7]
                except:
                    spindlestats[f+1,6] = 0
                    spindlestats[f+1,7] = 0
                    spindlestats[f+1,8] = 0
                    spindlestats[f+1,9] = 0
                    spindlestats[f+1,10] = 0

                try:
                    spindlestats[f+1,11] = spindle_df.iloc[2,0]
                    spindlestats[f+1,12] = spindle_df.iloc[2,1]
                    spindlestats[f+1,13] = spindle_df.iloc[2,2]
                    spindlestats[f+1,14] = spindle_df.iloc[2,3]
                    spindlestats[f+1,15] = spindle_df.iloc[2,7]
                except:
                    spindlestats[f+1,11] = 0
                    spindlestats[f+1,12] = 0
                    spindlestats[f+1,13] = 0
                    spindlestats[f+1,14] = 0
                    spindlestats[f+1,15] = 0

                try:
                    spindlestats[f+1,16] = spindle_df.iloc[3,0]
                    spindlestats[f+1,17] = spindle_df.iloc[3,1]
                    spindlestats[f+1,18] = spindle_df.iloc[3,2]
                    spindlestats[f+1,19] = spindle_df.iloc[3,3]
                    spindlestats[f+1,20] = spindle_df.iloc[3,7]
                except:
                    spindlestats[f+1,16] = 0
                    spindlestats[f+1,17] = 0
                    spindlestats[f+1,18] = 0
                    spindlestats[f+1,19] = 0
                    spindlestats[f+1,20] = 0


                try:
                    spindlestats[f+1,21] = spindle_df.iloc[6,0]
                    spindlestats[f+1,22] = spindle_df.iloc[6,1]
                    spindlestats[f+1,23] = spindle_df.iloc[6,2]
                    spindlestats[f+1,24] = spindle_df.iloc[6,3]
                    spindlestats[f+1,25] = spindle_df.iloc[6,7]

                    spindlestats[f+1,26] = spindle_df.iloc[7,0]
                    spindlestats[f+1,27] = spindle_df.iloc[7,1]
                    spindlestats[f+1,28] = spindle_df.iloc[7,2]
                    spindlestats[f+1,29] = spindle_df.iloc[7,3]
                    spindlestats[f+1,30] = spindle_df.iloc[7,7]

                    spindlestats[f+1,31] = spindle_df.iloc[8,0]
                    spindlestats[f+1,32] = spindle_df.iloc[8,1]
                    spindlestats[f+1,33] = spindle_df.iloc[8,2]
                    spindlestats[f+1,34] = spindle_df.iloc[8,3]
                    spindlestats[f+1,35] = spindle_df.iloc[8,7]

                    spindlestats[f+1,36] = spindle_df.iloc[9,0]
                    spindlestats[f+1,37] = spindle_df.iloc[9,1]
                    spindlestats[f+1,38] = spindle_df.iloc[9,2]
                    spindlestats[f+1,39] = spindle_df.iloc[9,3]
                    spindlestats[f+1,40] = spindle_df.iloc[9,7]

                except:
                    spindlestats[f+1,21] = 0
                    spindlestats[f+1,22] = 0
                    spindlestats[f+1,23] = 0
                    spindlestats[f+1,24] = 0
                    spindlestats[f+1,25] = 0

                    spindlestats[f+1,26] = 0
                    spindlestats[f+1,27] = 0
                    spindlestats[f+1,28] = 0
                    spindlestats[f+1,29] = 0
                    spindlestats[f+1,30] = 0

                    spindlestats[f+1,31] = 0
                    spindlestats[f+1,32] = 0
                    spindlestats[f+1,33] = 0
                    spindlestats[f+1,34] = 0
                    spindlestats[f+1,35] = 0

                    spindlestats[f+1,36] = 0
                    spindlestats[f+1,37] = 0
                    spindlestats[f+1,38] = 0
                    spindlestats[f+1,39] = 0
                    spindlestats[f+1,40] = 0
            
            except:
                spindlestats[f+1,1] = 0
                spindlestats[f+1,2] = 0
                spindlestats[f+1,3] = 0
                spindlestats[f+1,4] = 0
                spindlestats[f+1,5] = 0
                spindlestats[f+1,6] = 0
                spindlestats[f+1,7] = 0
                spindlestats[f+1,8] = 0
                spindlestats[f+1,9] = 0
                spindlestats[f+1,10] = 0
                spindlestats[f+1,11] = 0
                spindlestats[f+1,12] = 0
                spindlestats[f+1,13] = 0
                spindlestats[f+1,14] = 0
                spindlestats[f+1,15] = 0
                spindlestats[f+1,16] = 0
                spindlestats[f+1,17] = 0
                spindlestats[f+1,18] = 0
                spindlestats[f+1,19] = 0
                spindlestats[f+1,20] = 0
                spindlestats[f+1,21] = 0
                spindlestats[f+1,22] = 0
                spindlestats[f+1,23] = 0
                spindlestats[f+1,24] = 0
                spindlestats[f+1,25] = 0
                spindlestats[f+1,26] = 0
                spindlestats[f+1,27] = 0
                spindlestats[f+1,28] = 0
                spindlestats[f+1,29] = 0
                spindlestats[f+1,30] = 0
                spindlestats[f+1,31] = 0
                spindlestats[f+1,32] = 0
                spindlestats[f+1,33] = 0
                spindlestats[f+1,34] = 0
                spindlestats[f+1,35] = 0
                spindlestats[f+1,36] = 0
                spindlestats[f+1,37] = 0
                spindlestats[f+1,38] = 0
                spindlestats[f+1,39] = 0
                spindlestats[f+1,40] = 0
                
        
Spindle_outfile = open('/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/' + 'spindle stats.csv','w')
with Spindle_outfile:
   writer = csv.writer(Spindle_outfile,delimiter=',')
   writer.writerows(spindlestats)
    
    