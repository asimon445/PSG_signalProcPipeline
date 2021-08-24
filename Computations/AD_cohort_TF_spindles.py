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

COHORT = ['July2021']

for co in range(len(COHORT)):

    PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/AD_cohort/Preprocessed_fewerElecs/%s/' % (COHORT[co])

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
    
    failures = []

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



    #format output header  -- sprindles
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

        if (f < 10000):    #unindent not working, too lazy to delete this
            PSG_fnum = FILES[f][np.s_[pathsep+1:pathsep+6]]
            Hyp_fnum= HYPNO[f][np.s_[pathsep+1:pathsep+6]]

                # make sure that the participant numbers between the PSG file and the hypnogram file for iteration 'f' are the same
            if PSG_fnum == Hyp_fnum:

                eeg = mne.io.read_raw_fif(FILES[f], preload=True)
                
                eeg.pick_channels(['EEG F3-Ref','EEG F4-Ref','EEG C3-Ref','EEG C4-Ref','EEG O1-Ref','EEG O2-Ref'])  # Select a subset of EEG channels
                data = eeg.get_data() 
                data = data*1000000

                channels = eeg.ch_names
                info = eeg.info
                sfreq = eeg.info["sfreq"]

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

                if data.shape[1] == hypnogram.shape[0]:
#                     try:

#                         sp = yasa.spindles_detect(data, sf, ch_names=channels, hypno=hypnogram, include=(2, 3))

#                         # Get the full detection dataframe

#                         spindle_df = sp.summary(grp_chan=True, grp_stage=True, aggfunc='mean')

#                             #pull spindle density from stage 2 and 3 and store them
#                         spindlestats[f+1,0] = Hyp_fnum

#                         try:
#                             spindlestats[f+1,1] = spindle_df.iloc[0,0]
#                             spindlestats[f+1,2] = spindle_df.iloc[0,1]
#                             spindlestats[f+1,3] = spindle_df.iloc[0,2]
#                             spindlestats[f+1,4] = spindle_df.iloc[0,3]
#                             spindlestats[f+1,5] = spindle_df.iloc[0,7]
#                         except:
#                             spindlestats[f+1,1] = 0
#                             spindlestats[f+1,2] = 0
#                             spindlestats[f+1,3] = 0
#                             spindlestats[f+1,4] = 0
#                             spindlestats[f+1,5] = 0

#                         try:
#                             spindlestats[f+1,6] = spindle_df.iloc[1,0]
#                             spindlestats[f+1,7] = spindle_df.iloc[1,1]
#                             spindlestats[f+1,8] = spindle_df.iloc[1,2]
#                             spindlestats[f+1,9] = spindle_df.iloc[1,3]
#                             spindlestats[f+1,10] = spindle_df.iloc[1,7]
#                         except:
#                             spindlestats[f+1,6] = 0
#                             spindlestats[f+1,7] = 0
#                             spindlestats[f+1,8] = 0
#                             spindlestats[f+1,9] = 0
#                             spindlestats[f+1,10] = 0

#                         try:
#                             spindlestats[f+1,11] = spindle_df.iloc[2,0]
#                             spindlestats[f+1,12] = spindle_df.iloc[2,1]
#                             spindlestats[f+1,13] = spindle_df.iloc[2,2]
#                             spindlestats[f+1,14] = spindle_df.iloc[2,3]
#                             spindlestats[f+1,15] = spindle_df.iloc[2,7]
#                         except:
#                             spindlestats[f+1,11] = 0
#                             spindlestats[f+1,12] = 0
#                             spindlestats[f+1,13] = 0
#                             spindlestats[f+1,14] = 0
#                             spindlestats[f+1,15] = 0

#                         try:
#                             spindlestats[f+1,16] = spindle_df.iloc[3,0]
#                             spindlestats[f+1,17] = spindle_df.iloc[3,1]
#                             spindlestats[f+1,18] = spindle_df.iloc[3,2]
#                             spindlestats[f+1,19] = spindle_df.iloc[3,3]
#                             spindlestats[f+1,20] = spindle_df.iloc[3,7]
#                         except:
#                             spindlestats[f+1,16] = 0
#                             spindlestats[f+1,17] = 0
#                             spindlestats[f+1,18] = 0
#                             spindlestats[f+1,19] = 0
#                             spindlestats[f+1,20] = 0


#                         try:
#                             spindlestats[f+1,21] = spindle_df.iloc[6,0]
#                             spindlestats[f+1,22] = spindle_df.iloc[6,1]
#                             spindlestats[f+1,23] = spindle_df.iloc[6,2]
#                             spindlestats[f+1,24] = spindle_df.iloc[6,3]
#                             spindlestats[f+1,25] = spindle_df.iloc[6,7]

#                             spindlestats[f+1,26] = spindle_df.iloc[7,0]
#                             spindlestats[f+1,27] = spindle_df.iloc[7,1]
#                             spindlestats[f+1,28] = spindle_df.iloc[7,2]
#                             spindlestats[f+1,29] = spindle_df.iloc[7,3]
#                             spindlestats[f+1,30] = spindle_df.iloc[7,7]

#                             spindlestats[f+1,31] = spindle_df.iloc[8,0]
#                             spindlestats[f+1,32] = spindle_df.iloc[8,1]
#                             spindlestats[f+1,33] = spindle_df.iloc[8,2]
#                             spindlestats[f+1,34] = spindle_df.iloc[8,3]
#                             spindlestats[f+1,35] = spindle_df.iloc[8,7]

#                             spindlestats[f+1,36] = spindle_df.iloc[9,0]
#                             spindlestats[f+1,37] = spindle_df.iloc[9,1]
#                             spindlestats[f+1,38] = spindle_df.iloc[9,2]
#                             spindlestats[f+1,39] = spindle_df.iloc[9,3]
#                             spindlestats[f+1,40] = spindle_df.iloc[9,7]

#                         except:
#                             for c in range(spindlestats[0,:].shape[0]):
#                                 if c > 20:
#                                     spindlestats[f+1,c] = 0

#                     except:
#                         for c in range(spindlestats[0,:].shape[0]):
#                             spindlestats[f+1,c] = 0

            ################# compute band power #################
                    where_REM = np.isin(hypnogram, ['0'])  # find the stretches of EEG data that are in this specified stage
                    data_REM = data[:, where_REM]  
                    
                #this line is for a one time use (selects first 5 min of wake data), delete it next time I use this
                    data_REM = data_REM[:,0:30000]

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

                else:
                    failures.append(FILES[f])   #list files with sleep but without a hypnogram here. Save at end

#     SW_outfile = open(PATH + 'Slow wave density stats.csv','w')
#     with SW_outfile:
#        writer = csv.writer(SW_outfile,delimiter=',')
#        writer.writerows(SWstats)


    SpecPow_outfile = open(PATH + 'wake spectral stats.csv','w')
    with SpecPow_outfile:
        writer = csv.writer(SpecPow_outfile,delimiter=',')
        writer.writerows(PowStats)    


#     Spindle_outfile = open(PATH + 'spindle stats.csv','w')
#     with Spindle_outfile:
#        writer = csv.writer(Spindle_outfile,delimiter=',')
#        writer.writerows(spindlestats)
        
        #save failures
    fails_outfile = open(PATH + 'failures_timefreq.csv','w')
    with fails_outfile:
        writer = csv.writer(fails_outfile,delimiter=' ')
        writer.writerows(failures)