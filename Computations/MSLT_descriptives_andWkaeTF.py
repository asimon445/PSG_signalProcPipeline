import glob
import numpy as np
import mne
import yasa
from matplotlib import pyplot as plt
import matplotlib as mat
import os
import datetime
from scipy.special import erf
import csv
from scipy.stats import zscore
from scipy.signal import welch
import scipy.signal as signal
import seaborn as sns
import pandas as pd
sns.set(font_scale=1.2)

PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/Resting/preprocessed/EC_1p5_stdev_new/'

TFcompute = True   #compute time-freq stats

SUBDIRS = glob.glob(PATH + '/*/')   #list subdirectories
SUBDIRS.sort()

SF = 100    #Define frequency to downsample to -- Nyquist rate is 90 Hz for resolving spectral power in frequencies up to 45 Hz (gamma = 30-45 Hz -- per Walsh et al., 2017) 
            #VERY IMPORTANT THAT THE SAMPLING RATE MUST BE KEPT ABOVE THIS
            #Raw data is sampled at 400 Hz -- we don't need to downsample but can if we want to reduce size of data
        
#format output header  -- descriptives
Descriptives = np.zeros((len(SUBDIRS)+1,7), dtype=np.int)
Descriptives = Descriptives.astype('U30')

Descriptives[0,0] = 'Subject'
Descriptives[0,1] = 'Trials Slept'
Descriptives[0,2] = 'Total trials'
Descriptives[0,3] = '% trials slept'
Descriptives[0,4] = 'Minutes slept'
Descriptives[0,5] = 'Total minutes'
Descriptives[0,6] = '% minutes slept'


#format output header  -- time-freq stats
TimeF = np.zeros((len(SUBDIRS)+1,214), dtype=np.int)
TimeF = TimeF.astype('U30')
TimeF = np.full_like(TimeF, np.nan)

TimeF[0,0] = 'Subject'
TimeF[0,1] = 'T1 Frontal SO sleep'
TimeF[0,2] = 'T1 Frontal delta sleep'
TimeF[0,3] = 'T1 Frontal theta sleep'
TimeF[0,4] = 'T1 Frontal alpha sleep'
TimeF[0,5] = 'T1 Frontal sigma sleep'
TimeF[0,6] = 'T1 Frontal beta sleep'
TimeF[0,7] = 'T1 Frontal gamma sleep'

TimeF[0,8] = 'T2 Frontal SO sleep'
TimeF[0,9] = 'T2 Frontal delta sleep'
TimeF[0,10] = 'T2 Frontal theta sleep'
TimeF[0,11] = 'T2 Frontal alpha sleep'
TimeF[0,12] = 'T2 Frontal sigma sleep'
TimeF[0,13] = 'T2 Frontal beta sleep'
TimeF[0,14] = 'T2 Frontal gamma sleep'

TimeF[0,15] = 'T3 Frontal SO sleep'
TimeF[0,16] = 'T3 Frontal delta sleep'
TimeF[0,17] = 'T3 Frontal theta sleep'
TimeF[0,18] = 'T3 Frontal alpha sleep'
TimeF[0,19] = 'T3 Frontal sigma sleep'
TimeF[0,20] = 'T3 Frontal beta sleep'
TimeF[0,21] = 'T3 Frontal gamma sleep'

TimeF[0,22] = 'T4 Frontal SO sleep'
TimeF[0,23] = 'T4 Frontal delta sleep'
TimeF[0,24] = 'T4 Frontal theta sleep'
TimeF[0,25] = 'T4 Frontal alpha sleep'
TimeF[0,26] = 'T4 Frontal sigma sleep'
TimeF[0,27] = 'T4 Frontal beta sleep'
TimeF[0,28] = 'T4 Frontal gamma sleep'

TimeF[0,29] = 'T5 Frontal SO sleep'
TimeF[0,30] = 'T5 Frontal delta sleep'
TimeF[0,31] = 'T5 Frontal theta sleep'
TimeF[0,32] = 'T5 Frontal alpha sleep'
TimeF[0,33] = 'T5 Frontal sigma sleep'
TimeF[0,37] = 'T5 Frontal beta sleep'
TimeF[0,38] = 'T5 Frontal gamma sleep'


TimeF[0,39] = 'T1 Central SO sleep'
TimeF[0,40] = 'T1 Central delta sleep'
TimeF[0,41] = 'T1 Central theta sleep'
TimeF[0,42] = 'T1 Central alpha sleep'
TimeF[0,43] = 'T1 Central sigma sleep'
TimeF[0,44] = 'T1 Central beta sleep'
TimeF[0,45] = 'T1 Central gamma sleep'

TimeF[0,46] = 'T2 Central SO sleep'
TimeF[0,47] = 'T2 Central delta sleep'
TimeF[0,48] = 'T2 Central theta sleep'
TimeF[0,49] = 'T2 Central alpha sleep'
TimeF[0,50] = 'T2 Central sigma sleep'
TimeF[0,51] = 'T2 Central beta sleep'
TimeF[0,52] = 'T2 Central gamma sleep'

TimeF[0,53] = 'T3 Central SO sleep'
TimeF[0,54] = 'T3 Central delta sleep'
TimeF[0,55] = 'T3 Central theta sleep'
TimeF[0,56] = 'T3 Central alpha sleep'
TimeF[0,57] = 'T3 Central sigma sleep'
TimeF[0,58] = 'T3 Central beta sleep'
TimeF[0,59] = 'T3 Central gamma sleep'

TimeF[0,60] = 'T4 Central SO sleep'
TimeF[0,61] = 'T4 Central delta sleep'
TimeF[0,62] = 'T4 Central theta sleep'
TimeF[0,63] = 'T4 Central alpha sleep'
TimeF[0,64] = 'T4 Central sigma sleep'
TimeF[0,65] = 'T4 Central beta sleep'
TimeF[0,66] = 'T4 Central gamma sleep'

TimeF[0,67] = 'T5 Central SO sleep'
TimeF[0,68] = 'T5 Central delta sleep'
TimeF[0,69] = 'T5 Central theta sleep'
TimeF[0,70] = 'T5 Central alpha sleep'
TimeF[0,71] = 'T5 Central sigma sleep'
TimeF[0,72] = 'T5 Central beta sleep'
TimeF[0,73] = 'T5 Central gamma sleep'



TimeF[0,74] = 'T1 Posterior SO sleep'
TimeF[0,75] = 'T1 Posterior delta sleep'
TimeF[0,76] = 'T1 Posterior theta sleep'
TimeF[0,77] = 'T1 Posterior alpha sleep'
TimeF[0,78] = 'T1 Posterior sigma sleep'
TimeF[0,79] = 'T1 Posterior beta sleep'
TimeF[0,80] = 'T1 Posterior gamma sleep'

TimeF[0,81] = 'T2 Posterior SO sleep'
TimeF[0,82] = 'T2 Posterior delta sleep'
TimeF[0,83] = 'T2 Posterior theta sleep'
TimeF[0,84] = 'T2 Posterior alpha sleep'
TimeF[0,85] = 'T2 Posterior sigma sleep'
TimeF[0,86] = 'T2 Posterior beta sleep'
TimeF[0,87] = 'T2 Posterior gamma sleep'

TimeF[0,88] = 'T3 Posterior SO sleep'
TimeF[0,89] = 'T3 Posterior delta sleep'
TimeF[0,90] = 'T3 Posterior theta sleep'
TimeF[0,91] = 'T3 Posterior alpha sleep'
TimeF[0,92] = 'T3 Posterior sigma sleep'
TimeF[0,93] = 'T3 Posterior beta sleep'
TimeF[0,94] = 'T3 Posterior gamma sleep'

TimeF[0,95] = 'T4 Posterior SO sleep'
TimeF[0,96] = 'T4 Posterior delta sleep'
TimeF[0,97] = 'T4 Posterior theta sleep'
TimeF[0,98] = 'T4 Posterior alpha sleep'
TimeF[0,99] = 'T4 Posterior sigma sleep'
TimeF[0,100] = 'T4 Posterior beta sleep'
TimeF[0,101] = 'T4 Posterior gamma sleep'

TimeF[0,102] = 'T5 Posterior SO sleep'
TimeF[0,103] = 'T5 Posterior delta sleep'
TimeF[0,104] = 'T5 Posterior theta sleep'
TimeF[0,105] = 'T5 Posterior alpha sleep'
TimeF[0,106] = 'T5 Posterior sigma sleep'
TimeF[0,107] = 'T5 Posterior beta sleep'
TimeF[0,108] = 'T5 Posterior gamma sleep'


TimeF[0,109] = 'T1 Frontal SO wake'
TimeF[0,110] = 'T1 Frontal delta wake'
TimeF[0,111] = 'T1 Frontal theta wake'
TimeF[0,112] = 'T1 Frontal alpha wake'
TimeF[0,113] = 'T1 Frontal sigma wake'
TimeF[0,114] = 'T1 Frontal beta wake'
TimeF[0,115] = 'T1 Frontal gamma wake'

TimeF[0,116] = 'T2 Frontal SO wake'
TimeF[0,117] = 'T2 Frontal delta wake'
TimeF[0,118] = 'T2 Frontal theta wake'
TimeF[0,119] = 'T2 Frontal alpha wake'
TimeF[0,120] = 'T2 Frontal sigma wake'
TimeF[0,121] = 'T2 Frontal beta wake'
TimeF[0,122] = 'T2 Frontal gamma wake'

TimeF[0,123] = 'T3 Frontal SO wake'
TimeF[0,124] = 'T3 Frontal delta wake'
TimeF[0,125] = 'T3 Frontal theta wake'
TimeF[0,126] = 'T3 Frontal alpha wake'
TimeF[0,127] = 'T3 Frontal sigma wake'
TimeF[0,128] = 'T3 Frontal beta wake'
TimeF[0,129] = 'T3 Frontal gamma wake'

TimeF[0,130] = 'T4 Frontal SO wake'
TimeF[0,131] = 'T4 Frontal delta wake'
TimeF[0,132] = 'T4 Frontal theta wake'
TimeF[0,133] = 'T4 Frontal alpha wake'
TimeF[0,134] = 'T4 Frontal sigma wake'
TimeF[0,135] = 'T4 Frontal beta wake'
TimeF[0,136] = 'T4 Frontal gamma wake'

TimeF[0,137] = 'T5 Frontal SO wake'
TimeF[0,138] = 'T5 Frontal delta wake'
TimeF[0,139] = 'T5 Frontal theta wake'
TimeF[0,140] = 'T5 Frontal alpha wake'
TimeF[0,141] = 'T5 Frontal sigma wake'
TimeF[0,142] = 'T5 Frontal beta wake'
TimeF[0,143] = 'T5 Frontal gamma wake'


TimeF[0,144] = 'T1 Central SO wake'
TimeF[0,145] = 'T1 Central delta wake'
TimeF[0,146] = 'T1 Central theta wake'
TimeF[0,147] = 'T1 Central alpha wake'
TimeF[0,148] = 'T1 Central sigma wake'
TimeF[0,149] = 'T1 Central beta wake'
TimeF[0,150] = 'T1 Central gamma wake'

TimeF[0,151] = 'T2 Central SO wake'
TimeF[0,152] = 'T2 Central delta wake'
TimeF[0,153] = 'T2 Central theta wake'
TimeF[0,154] = 'T2 Central alpha wake'
TimeF[0,155] = 'T2 Central sigma wake'
TimeF[0,156] = 'T2 Central beta wake'
TimeF[0,157] = 'T2 Central gamma wake'

TimeF[0,158] = 'T3 Central SO wake'
TimeF[0,159] = 'T3 Central delta wake'
TimeF[0,160] = 'T3 Central theta wake'
TimeF[0,161] = 'T3 Central alpha wake'
TimeF[0,162] = 'T3 Central sigma wake'
TimeF[0,163] = 'T3 Central beta wake'
TimeF[0,164] = 'T3 Central gamma wake'

TimeF[0,165] = 'T4 Central SO wake'
TimeF[0,166] = 'T4 Central delta wake'
TimeF[0,167] = 'T4 Central theta wake'
TimeF[0,168] = 'T4 Central alpha wake'
TimeF[0,169] = 'T4 Central sigma wake'
TimeF[0,170] = 'T4 Central beta wake'
TimeF[0,171] = 'T4 Central gamma wake'

TimeF[0,172] = 'T5 Central SO wake'
TimeF[0,173] = 'T5 Central delta wake'
TimeF[0,174] = 'T5 Central theta wake'
TimeF[0,175] = 'T5 Central alpha wake'
TimeF[0,176] = 'T5 Central sigma wake'
TimeF[0,177] = 'T5 Central beta wake'
TimeF[0,178] = 'T5 Central gamma wake'



TimeF[0,179] = 'T1 Posterior SO wake'
TimeF[0,180] = 'T1 Posterior delta wake'
TimeF[0,181] = 'T1 Posterior theta wake'
TimeF[0,182] = 'T1 Posterior alpha wake'
TimeF[0,183] = 'T1 Posterior sigma wake'
TimeF[0,184] = 'T1 Posterior beta wake'
TimeF[0,185] = 'T1 Posterior gamma wake'

TimeF[0,186] = 'T2 Posterior SO wake'
TimeF[0,187] = 'T2 Posterior delta wake'
TimeF[0,188] = 'T2 Posterior theta wake'
TimeF[0,189] = 'T2 Posterior alpha wake'
TimeF[0,190] = 'T2 Posterior sigma wake'
TimeF[0,191] = 'T2 Posterior beta wake'
TimeF[0,192] = 'T2 Posterior gamma wake'

TimeF[0,193] = 'T3 Posterior SO wake'
TimeF[0,194] = 'T3 Posterior delta wake'
TimeF[0,195] = 'T3 Posterior theta wake'
TimeF[0,196] = 'T3 Posterior alpha wake'
TimeF[0,197] = 'T3 Posterior sigma wake'
TimeF[0,198] = 'T3 Posterior beta wake'
TimeF[0,199] = 'T3 Posterior gamma wake'

TimeF[0,200] = 'T4 Posterior SO wake'
TimeF[0,201] = 'T4 Posterior delta wake'
TimeF[0,202] = 'T4 Posterior theta wake'
TimeF[0,203] = 'T4 Posterior alpha wake'
TimeF[0,204] = 'T4 Posterior sigma wake'
TimeF[0,205] = 'T4 Posterior beta wake'
TimeF[0,206] = 'T4 Posterior gamma wake'

TimeF[0,207] = 'T5 Posterior SO wake'
TimeF[0,208] = 'T5 Posterior delta wake'
TimeF[0,209] = 'T5 Posterior theta wake'
TimeF[0,210] = 'T5 Posterior alpha wake'
TimeF[0,211] = 'T5 Posterior sigma wake'
TimeF[0,212] = 'T5 Posterior beta wake'
TimeF[0,213] = 'T5 Posterior gamma wake'




for s in range(len(SUBDIRS)):

    # this is for indexing the last '/' in the path so that we can pull the filename easily for each participant
    sep = '/'
    currdir = SUBDIRS[s]
    def find(currdir, sep):
        return [i for i, ltr in enumerate(currdir) if ltr == sep]

    pathseps = list(find(currdir,sep))
    pathsep = pathseps[len(pathseps)-1]   #this here is the position of the final '/'
    
    # generate lists of .edf and .xls files
    fifs = glob.glob(currdir + '*.fif')
    fifs.sort()
    
    xlss = glob.glob(currdir + '*.csv')
    xlss.sort()
    
    ix_slept_tr = 0

    for f in range(len(fifs)):
        
        isT1 = fifs[f].find('T1')
        isT2 = fifs[f].find('T2')
        isT3 = fifs[f].find('T3')
        isT4 = fifs[f].find('T4')
        isT5 = fifs[f].find('T5')

        xl_match = -1   #this will index the position of the corresponding hypnogram in the hypnogram list for the edf in this iteration of the loop
        if isT1 > 0:

            for xl in range(len(xlss)):

                isT1too = xlss[xl].find('T1')

                if isT1too > 0:  #we have a match!

                    xl_match = xl

        elif isT2 > 0:

            for xl in range(len(xlss)):

                isT2too = xlss[xl].find('T2')

                if isT2too > 0:  #we have a match!

                    xl_match = xl

        elif isT3 > 0:

            for xl in range(len(xlss)):

                isT3too = xlss[xl].find('T3')

                if isT3too > 0:  #we have a match!

                    xl_match = xl

        elif isT4 > 0:

            for xl in range(len(xlss)):

                isT4too = xlss[xl].find('T4')

                if isT4too > 0:  #we have a match!

                    xl_match = xl

        elif isT5 > 0:

            for xl in range(len(xlss)):

                isT5too = xlss[xl].find('T5')

                if isT5too > 0:  #we have a match!

                    xl_match = xl

        else:
            xl_match = -1   
        #if there is not a hypnogram for this trial (xl_match == -1), then document this subject and move on.
        if xl_match >= 0:

            subpathseps = list(find(SUBDIRS[s],sep))
            subpathsep = subpathseps[len(pathseps)-1]   #this here is the position of the final '/'
                
            Hyp_fname = xlss[xl_match][np.s_[pathsep+1:len(xlss[0])]]
            PSG_fname = fifs[f][np.s_[pathsep+1:len(fifs[0])]]
            
            PSG_fnum = fifs[f][np.s_[pathsep+1:pathsep+6]]
            
            #load PSG file
            eeg = mne.io.read_raw_fif(fifs[f], preload=True)
            
            #store EEG data into an e x t matrix, where e = num elecs and t = samples
            data = eeg.get_data() 
            
            data = data*1000000

            channels = eeg.ch_names

            times = np.arange(data.size) / SF
            
            hypnogram = []
            hypnogram = np.loadtxt(fname = xlss[xl_match],dtype = 'str',delimiter = ',',skiprows=0) 
            hypnogram.astype('U8')
            
            hypno = np.zeros((len(hypnogram)+1), dtype=np.float)
            
            #convert some of the odd data types to something I can work with
            for i in range(len(hypnogram)):
            
                if hypnogram[i] == '0 . 0':
                    
                    hypno[i] = 0
                    
                elif hypnogram[i] == '1 . 0':
                    
                    hypno[i] = 1
                    
                elif hypnogram[i] == '- 1':
                    
                    hypno[i] = -1
            
                elif hypnogram[i] == '2 . 0':
                    
                    hypno[i] = 2
                
                elif hypnogram[i] == '3 . 0':
                    
                    hypno[i] = 3
                    
                elif hypnogram[i] == '4 . 0':
                    
                    hypno[i] = 4
                
            for i in range(len(hypnogram)):
            
                if hypnogram[i] == '0':
                    
                    hypno[i] = 0
                    
                elif hypnogram[i] == '1':
                    
                    hypno[i] = 1
                    
                elif hypnogram[i] == '- 1':
                    
                    hypno[i] = -1
            
                elif hypnogram[i] == '-1':
                    
                    hypno[i] = -1
                    
                elif hypnogram[i] == '-':
                
                    hypno[i] = -1
                    
                elif hypnogram[i] == '2 . 0':
                    
                    hypno[i] = 2
                
                elif hypnogram[i] == '3 . 0':
                    
                    hypno[i] = 3
                    
                elif hypnogram[i] == '4 . 0':
                    
                    hypno[i] = 4
            
            #like a dummy, I added 30 sec of garbage to the end of some hypnograms. Remove this. 
            if hypno.shape[0] > data.shape[1]:
                
                hypno = np.delete(hypno,np.s_[data.shape[1]:hypno.shape[0]],0)
                
                
                
            ix_1 = 0
            ix_2 = 0
            ix_3 = 0
            ix_4 = 0
            
            for h in range(hypno.shape[0]):
                
                if hypno[h] == 1.:   #count the number of samples when the participant was in stage 1, 2, 3, and REM
                    
                    ix_1 = ix_1 + 1
                    
                elif hypno[h] == 2.:
                    
                    ix_2 = ix_2 + 1
                    
                elif hypno[h] == 3.:
                    
                    ix_3 = ix_3 + 1
                    
                elif hypno[h] == 4.:
                    
                    ix_4 = ix_4 + 1
                    
            
            TimeF[s+1,0] = PSG_fnum
            if TFcompute:
                
                if isT1 > 0:
                    
                    hypno = hypno.astype('str')
                    
                    if ix_1 + ix_2 + ix_3 + ix_4 > 0:   #they slept
                        
                        #compute band power
                        where_sleep = np.isin(hypno, ['1.0','2.0','3.0','4.0'])  # True if sample is in N2 / N3, False otherwise
                        data_sleep = data[:, where_sleep]        

                        win = int(4 * SF)  # Window size is set to 4 seconds
                        freqs, psd = welch(data_sleep, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                        pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                        
                        # frontal channels, sleep
                        TimeF[s+1,1] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                        TimeF[s+1,2] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                        TimeF[s+1,3] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                        TimeF[s+1,4] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                        TimeF[s+1,5] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                        TimeF[s+1,6] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                        TimeF[s+1,7] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                        
                        # central channels, sleep
                        TimeF[s+1,39] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                        TimeF[s+1,40] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                        TimeF[s+1,41] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                        TimeF[s+1,42] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                        TimeF[s+1,43] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                        TimeF[s+1,44] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                        TimeF[s+1,45] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                        # posterior channels, sleep
                        TimeF[s+1,74] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                        TimeF[s+1,75] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                        TimeF[s+1,76] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                        TimeF[s+1,77] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                        TimeF[s+1,78] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                        TimeF[s+1,79] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                        TimeF[s+1,80] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                    
                    # frontal channels, wake
                    TimeF[s+1,109] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                    TimeF[s+1,110] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                    TimeF[s+1,111] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                    TimeF[s+1,112] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                    TimeF[s+1,113] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                    TimeF[s+1,114] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                    TimeF[s+1,115] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                    
                    # central channels, wake
                    TimeF[s+1,144] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                    TimeF[s+1,145] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                    TimeF[s+1,146] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                    TimeF[s+1,147] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                    TimeF[s+1,148] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                    TimeF[s+1,149] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                    TimeF[s+1,150] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                    # posterior channels, wake
                    TimeF[s+1,179] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                    TimeF[s+1,180] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                    TimeF[s+1,181] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                    TimeF[s+1,182] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                    TimeF[s+1,183] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                    TimeF[s+1,184] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                    TimeF[s+1,185] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2
                    
                    
                elif isT2 > 0:
                    
                    hypno = hypno.astype('str')
                    
                    if ix_1 + ix_2 + ix_3 + ix_4 > 0:   #they slept
                        
                        #compute band power
                        where_sleep = np.isin(hypno, ['1.0','2.0','3.0','4.0'])  # True if sample is in N2 / N3, False otherwise
                        data_sleep = data[:, where_sleep]        

                        win = int(4 * SF)  # Window size is set to 4 seconds
                        freqs, psd = welch(data_sleep, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                        pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                        
                        # frontal channels, sleep
                        TimeF[s+1,8] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                        TimeF[s+1,9] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                        TimeF[s+1,10] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                        TimeF[s+1,11] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                        TimeF[s+1,12] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                        TimeF[s+1,13] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                        TimeF[s+1,14] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                        
                        # central channels, sleep
                        TimeF[s+1,46] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                        TimeF[s+1,47] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                        TimeF[s+1,48] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                        TimeF[s+1,49] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                        TimeF[s+1,50] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                        TimeF[s+1,51] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                        TimeF[s+1,52] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                        # posterior channels, sleep
                        TimeF[s+1,81] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                        TimeF[s+1,82] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                        TimeF[s+1,83] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                        TimeF[s+1,84] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                        TimeF[s+1,85] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                        TimeF[s+1,86] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                        TimeF[s+1,87] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                    
                    # frontal channels, wake
                    TimeF[s+1,116] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                    TimeF[s+1,117] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                    TimeF[s+1,118] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                    TimeF[s+1,119] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                    TimeF[s+1,120] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                    TimeF[s+1,121] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                    TimeF[s+1,122] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                    
                    # central channels, wake
                    TimeF[s+1,151] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                    TimeF[s+1,152] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                    TimeF[s+1,153] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                    TimeF[s+1,154] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                    TimeF[s+1,155] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                    TimeF[s+1,156] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                    TimeF[s+1,157] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                    # posterior channels, wake
                    TimeF[s+1,186] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                    TimeF[s+1,187] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                    TimeF[s+1,188] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                    TimeF[s+1,189] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                    TimeF[s+1,190] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                    TimeF[s+1,191] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                    TimeF[s+1,192] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2
                    
                    
                elif isT3 > 0:
                    
                    hypno = hypno.astype('str')
                    
                    if ix_1 + ix_2 + ix_3 + ix_4 > 0:   #they slept
                        
                        #compute band power
                        where_sleep = np.isin(hypno, ['1.0','2.0','3.0','4.0'])  # True if sample is in N2 / N3, False otherwise
                        data_sleep = data[:, where_sleep]        

                        win = int(4 * SF)  # Window size is set to 4 seconds
                        freqs, psd = welch(data_sleep, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                        pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                        
                        # frontal channels, sleep
                        TimeF[s+1,15] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                        TimeF[s+1,16] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                        TimeF[s+1,17] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                        TimeF[s+1,18] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                        TimeF[s+1,19] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                        TimeF[s+1,20] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                        TimeF[s+1,21] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                        
                        # central channels, sleep
                        TimeF[s+1,53] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                        TimeF[s+1,54] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                        TimeF[s+1,55] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                        TimeF[s+1,56] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                        TimeF[s+1,57] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                        TimeF[s+1,58] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                        TimeF[s+1,59] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                        # posterior channels, sleep
                        TimeF[s+1,88] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                        TimeF[s+1,89] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                        TimeF[s+1,90] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                        TimeF[s+1,91] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                        TimeF[s+1,92] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                        TimeF[s+1,93] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                        TimeF[s+1,94] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                    
                    # frontal channels, wake
                    TimeF[s+1,123] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                    TimeF[s+1,124] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                    TimeF[s+1,125] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                    TimeF[s+1,126] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                    TimeF[s+1,127] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                    TimeF[s+1,128] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                    TimeF[s+1,129] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                    
                    # central channels, wake
                    TimeF[s+1,158] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                    TimeF[s+1,159] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                    TimeF[s+1,160] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                    TimeF[s+1,161] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                    TimeF[s+1,162] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                    TimeF[s+1,163] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                    TimeF[s+1,164] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                    # posterior channels, wake
                    TimeF[s+1,193] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                    TimeF[s+1,194] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                    TimeF[s+1,195] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                    TimeF[s+1,196] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                    TimeF[s+1,197] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                    TimeF[s+1,198] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                    TimeF[s+1,199] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2
                    
                    
                elif isT4 > 0:
                    
                    hypno = hypno.astype('str')
                    
                    if ix_1 + ix_2 + ix_3 + ix_4 > 0:   #they slept
                        
                        #compute band power
                        where_sleep = np.isin(hypno, ['1.0','2.0','3.0','4.0'])  # True if sample is in N2 / N3, False otherwise
                        data_sleep = data[:, where_sleep]        

                        win = int(4 * SF)  # Window size is set to 4 seconds
                        freqs, psd = welch(data_sleep, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                        pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                        
                        # frontal channels, sleep
                        TimeF[s+1,22] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                        TimeF[s+1,23] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                        TimeF[s+1,24] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                        TimeF[s+1,25] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                        TimeF[s+1,26] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                        TimeF[s+1,27] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                        TimeF[s+1,28] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                        
                        # central channels, sleep
                        TimeF[s+1,60] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                        TimeF[s+1,61] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                        TimeF[s+1,62] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                        TimeF[s+1,63] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                        TimeF[s+1,64] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                        TimeF[s+1,65] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                        TimeF[s+1,66] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                        # posterior channels, sleep
                        TimeF[s+1,95] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                        TimeF[s+1,96] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                        TimeF[s+1,97] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                        TimeF[s+1,98] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                        TimeF[s+1,99] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                        TimeF[s+1,100] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                        TimeF[s+1,101] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                    
                    # frontal channels, wake
                    TimeF[s+1,130] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                    TimeF[s+1,131] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                    TimeF[s+1,132] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                    TimeF[s+1,133] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                    TimeF[s+1,134] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                    TimeF[s+1,135] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                    TimeF[s+1,136] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                    
                    # central channels, wake
                    TimeF[s+1,165] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                    TimeF[s+1,166] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                    TimeF[s+1,167] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                    TimeF[s+1,168] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                    TimeF[s+1,169] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                    TimeF[s+1,170] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                    TimeF[s+1,171] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                    # posterior channels, wake
                    TimeF[s+1,200] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                    TimeF[s+1,201] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                    TimeF[s+1,202] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                    TimeF[s+1,203] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                    TimeF[s+1,204] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                    TimeF[s+1,205] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                    TimeF[s+1,206] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2
                    
                    
                elif isT5 > 0:
                
                    hypno = hypno.astype('str')
                    
                    if ix_1 + ix_2 + ix_3 + ix_4 > 0:   #they slept
                        
                        #compute band power
                        where_sleep = np.isin(hypno, ['1.0','2.0','3.0','4.0'])  # True if sample is in N2 / N3, False otherwise
                        data_sleep = data[:, where_sleep]        

                        win = int(4 * SF)  # Window size is set to 4 seconds
                        freqs, psd = welch(data_sleep, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                        pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                        
                        # frontal channels, sleep
                        TimeF[s+1,29] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                        TimeF[s+1,30] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                        TimeF[s+1,31] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                        TimeF[s+1,32] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                        TimeF[s+1,33] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                        TimeF[s+1,34] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                        TimeF[s+1,35] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                        
                        # central channels, sleep
                        TimeF[s+1,67] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                        TimeF[s+1,68] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                        TimeF[s+1,69] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                        TimeF[s+1,70] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                        TimeF[s+1,71] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                        TimeF[s+1,72] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                        TimeF[s+1,73] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                        # posterior channels, sleep
                        TimeF[s+1,102] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                        TimeF[s+1,103] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                        TimeF[s+1,104] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                        TimeF[s+1,105] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                        TimeF[s+1,106] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                        TimeF[s+1,107] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                        TimeF[s+1,108] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    pow_df = yasa.bandpower_from_psd(psd, freqs, ch_names=channels, bands=[(0.5, 1, 'LF SW'),(1, 4, 'Delta'),(4, 8, 'theta'),(8, 12, 'alpha'),(12, 15, 'sigma'),(15, 30, 'beta'),(30, 50, 'gamma')], relative=False)
                    
                    # frontal channels, wake
                    TimeF[s+1,137] = (pow_df.iloc[0,1] + pow_df.iloc[1,1]) / 2
                    TimeF[s+1,138] = (pow_df.iloc[0,2] + pow_df.iloc[1,2]) / 2
                    TimeF[s+1,139] = (pow_df.iloc[0,3] + pow_df.iloc[1,3]) / 2
                    TimeF[s+1,140] = (pow_df.iloc[0,4] + pow_df.iloc[1,4]) / 2
                    TimeF[s+1,141] = (pow_df.iloc[0,5] + pow_df.iloc[1,5]) / 2
                    TimeF[s+1,142] = (pow_df.iloc[0,6] + pow_df.iloc[1,6]) / 2
                    TimeF[s+1,143] = (pow_df.iloc[0,7] + pow_df.iloc[1,7]) / 2
                    
                    # central channels, wake
                    TimeF[s+1,172] = (pow_df.iloc[2,1] + pow_df.iloc[3,1]) / 2
                    TimeF[s+1,173] = (pow_df.iloc[2,2] + pow_df.iloc[3,2]) / 2
                    TimeF[s+1,174] = (pow_df.iloc[2,3] + pow_df.iloc[3,3]) / 2
                    TimeF[s+1,175] = (pow_df.iloc[2,4] + pow_df.iloc[3,4]) / 2
                    TimeF[s+1,176] = (pow_df.iloc[2,5] + pow_df.iloc[3,5]) / 2
                    TimeF[s+1,177] = (pow_df.iloc[2,6] + pow_df.iloc[3,6]) / 2
                    TimeF[s+1,178] = (pow_df.iloc[2,7] + pow_df.iloc[3,7]) / 2
                        
                    # posterior channels, wake
                    TimeF[s+1,207] = (pow_df.iloc[4,1] + pow_df.iloc[4,1]) / 2
                    TimeF[s+1,208] = (pow_df.iloc[4,2] + pow_df.iloc[4,2]) / 2
                    TimeF[s+1,209] = (pow_df.iloc[4,3] + pow_df.iloc[4,3]) / 2
                    TimeF[s+1,210] = (pow_df.iloc[4,4] + pow_df.iloc[4,4]) / 2
                    TimeF[s+1,211] = (pow_df.iloc[4,5] + pow_df.iloc[4,5]) / 2
                    TimeF[s+1,212] = (pow_df.iloc[4,6] + pow_df.iloc[4,6]) / 2
                    TimeF[s+1,213] = (pow_df.iloc[4,7] + pow_df.iloc[4,7]) / 2
            
                    
            if ix_1 > 0:   #this is an instance when the pt fell asleep. document it for descriptives
                ix_slept_tr = ix_slept_tr + 1
            elif ix_2 > 0:
                ix_slept_tr = ix_slept_tr + 1
            elif ix_3 > 0:
                ix_slept_tr = ix_slept_tr + 1
            elif ix_4 > 0:
                ix_slept_tr = ix_slept_tr + 1
                
           # count total amount of time slept
            if f == 0:
                tot_slept = ix_1 + ix_2 + ix_3 + ix_4
                tot_time = hypnogram.shape[0]
            else:
                tot_slept = tot_slept + ix_1 + ix_2 + ix_3 + ix_4
                tot_time = tot_time + hypnogram.shape[0]
                
    Descriptives[s+1,0] = PSG_fnum
    Descriptives[s+1,1] = ix_slept_tr
    Descriptives[s+1,2] = len(fifs)
    Descriptives[s+1,3] = (ix_slept_tr/len(fifs))*100
    Descriptives[s+1,4] = tot_slept/(SF*60)
    Descriptives[s+1,5] = tot_time/(SF*60)
    Descriptives[s+1,6] = (tot_slept/tot_time)*100


# Descriptives_outfile = open(PATH + 'MSLT_descriptives.csv','w')
# with Descriptives_outfile:
#     writer = csv.writer(Descriptives_outfile,delimiter=',')
#     writer.writerows(Descriptives)


MSLT_outfile = open(PATH + 'MSLT_TimeFreq.csv','w')
with MSLT_outfile:
    writer = csv.writer(MSLT_outfile,delimiter=',')
    writer.writerows(TimeF)

