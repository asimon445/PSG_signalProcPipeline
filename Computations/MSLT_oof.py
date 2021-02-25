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
from statistics import mean

PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/MSLT/preprocessed_1sec_windows_1point5_USEDTHISONE/'

TFcompute = True   #compute time-freq stats

SUBDIRS = glob.glob(PATH + '/*/')   #list subdirectories
SUBDIRS.sort()

SF = 100    #Define frequency to downsample to -- Nyquist rate is 90 Hz for resolving spectral power in frequencies up to 45 Hz (gamma = 30-45 Hz -- per Walsh et al., 2017) 
            #VERY IMPORTANT THAT THE SAMPLING RATE MUST BE KEPT ABOVE THIS
            #Raw data is sampled at 400 Hz -- we don't need to downsample but can if we want to reduce size of data
        

#format output header  -- time-freq stats
TimeF = np.zeros((len(SUBDIRS)+1,31), dtype=np.int)
TimeF = TimeF.astype('U30')
TimeF = np.full_like(TimeF, np.nan)

TimeF[0,0] = 'Subject'
TimeF[0,1] = 'T1 frontal sleep'
TimeF[0,2] = 'T1 central sleep'
TimeF[0,3] = 'T1 posterior sleep'
TimeF[0,4] = 'T2 frontal sleep'
TimeF[0,5] = 'T2 central sleep'
TimeF[0,6] = 'T2 posterior sleep'
TimeF[0,7] = 'T3 frontal sleep'
TimeF[0,8] = 'T3 central sleep'
TimeF[0,9] = 'T3 posterior sleep'
TimeF[0,10] = 'T4 frontal sleep'
TimeF[0,11] = 'T4 central sleep'
TimeF[0,12] = 'T4 posterior sleep'
TimeF[0,13] = 'T5 frontal sleep'
TimeF[0,14] = 'T5 central sleep'
TimeF[0,15] = 'T5 posterior sleep'
TimeF[0,16] = 'T1 frontal wake'
TimeF[0,17] = 'T1 central wake'
TimeF[0,18] = 'T1 posterior wake'
TimeF[0,19] = 'T2 frontal wake'
TimeF[0,20] = 'T2 central wake'
TimeF[0,21] = 'T2 posterior wake'
TimeF[0,22] = 'T3 frontal wake'
TimeF[0,23] = 'T3 central wake'
TimeF[0,24] = 'T3 posterior wake'
TimeF[0,25] = 'T4 frontal wake'
TimeF[0,26] = 'T4 central wake'
TimeF[0,27] = 'T4 posterior wake'
TimeF[0,28] = 'T5 frontal wake'
TimeF[0,29] = 'T5 central wake'
TimeF[0,30] = 'T5 posterior wake'


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
                        oof = np.zeros(len(data), dtype=np.float64)
                        for e, elec in enumerate(data):
                            xs = np.array(psd[e,:], dtype=np.float64)
                            ys = np.array(freqs, dtype=np.float64)

                            def best_fit_slope(xs,ys):
                                m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                     ((mean(xs)**2) - mean(xs**2)))
                                return m

                            oof[e] = best_fit_slope(xs,ys)

                        
                        TimeF[s+1,1] = (oof[0] + oof[1]) / 2  # frontal channels, sleep
                        TimeF[s+1,2] = (oof[2] + oof[3]) / 2  # central channels, sleep
                        TimeF[s+1,3] = (oof[4] + oof[5]) / 2  # posterior channels, sleep
                        

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz
                    oof = np.zeros(len(data), dtype=np.float64)
                    for e, elec in enumerate(data):
                        xs = np.array(psd[e,:], dtype=np.float64)
                        ys = np.array(freqs, dtype=np.float64)

                        def best_fit_slope(xs,ys):
                            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                    ((mean(xs)**2) - mean(xs**2)))
                            return m

                        oof[e] = best_fit_slope(xs,ys)
                        
                    #wake
                    TimeF[s+1,16] = (oof[0] + oof[1]) / 2  # frontal channels
                    TimeF[s+1,17] = (oof[2] + oof[3]) / 2  # central channels
                    TimeF[s+1,18] = (oof[4] + oof[5]) / 2  # posterior channels
                    
                    
                elif isT2 > 0:
                    
                    hypno = hypno.astype('str')
                    
                    if ix_1 + ix_2 + ix_3 + ix_4 > 0:   #they slept
                        
                        #compute band power
                        where_sleep = np.isin(hypno, ['1.0','2.0','3.0','4.0'])  # True if sample is in N2 / N3, False otherwise
                        data_sleep = data[:, where_sleep]        

                        win = int(4 * SF)  # Window size is set to 4 seconds
                        freqs, psd = welch(data_sleep, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                        oof = np.zeros(len(data), dtype=np.float64)
                        for e, elec in enumerate(data):
                            xs = np.array(psd[e,:], dtype=np.float64)
                            ys = np.array(freqs, dtype=np.float64)

                            def best_fit_slope(xs,ys):
                                m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                        ((mean(xs)**2) - mean(xs**2)))
                                return m

                            oof[e] = best_fit_slope(xs,ys)

                        #sleep
                        TimeF[s+1,4] = (oof[0] + oof[1]) / 2  # frontal channels
                        TimeF[s+1,5] = (oof[2] + oof[3]) / 2  # central channels
                        TimeF[s+1,6] = (oof[4] + oof[5]) / 2  # posterior channels
                        

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    oof = np.zeros(len(data), dtype=np.float64)
                    for e, elec in enumerate(data):
                        xs = np.array(psd[e,:], dtype=np.float64)
                        ys = np.array(freqs, dtype=np.float64)

                        def best_fit_slope(xs,ys):
                            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                    ((mean(xs)**2) - mean(xs**2)))
                            return m

                        oof[e] = best_fit_slope(xs,ys)
                        
                    #wake
                    TimeF[s+1,19] = (oof[0] + oof[1]) / 2  # frontal channels
                    TimeF[s+1,20] = (oof[2] + oof[3]) / 2  # central channels
                    TimeF[s+1,21] = (oof[4] + oof[5]) / 2  # posterior channels
                    
                    
                    
                elif isT3 > 0:
                    
                    hypno = hypno.astype('str')
                    
                    if ix_1 + ix_2 + ix_3 + ix_4 > 0:   #they slept
                        
                        #compute band power
                        where_sleep = np.isin(hypno, ['1.0','2.0','3.0','4.0'])  # True if sample is in N2 / N3, False otherwise
                        data_sleep = data[:, where_sleep]        

                        win = int(4 * SF)  # Window size is set to 4 seconds
                        freqs, psd = welch(data_sleep, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                        oof = np.zeros(len(data), dtype=np.float64)
                        for e, elec in enumerate(data):
                            xs = np.array(psd[e,:], dtype=np.float64)
                            ys = np.array(freqs, dtype=np.float64)

                            def best_fit_slope(xs,ys):
                                m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                        ((mean(xs)**2) - mean(xs**2)))
                                return m

                            oof[e] = best_fit_slope(xs,ys)

                        #sleep
                        TimeF[s+1,7] = (oof[0] + oof[1]) / 2  # frontal channels
                        TimeF[s+1,8] = (oof[2] + oof[3]) / 2  # central channels
                        TimeF[s+1,9] = (oof[4] + oof[5]) / 2  # posterior channels
                        

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    oof = np.zeros(len(data), dtype=np.float64)
                    for e, elec in enumerate(data):
                        xs = np.array(psd[e,:], dtype=np.float64)
                        ys = np.array(freqs, dtype=np.float64)

                        def best_fit_slope(xs,ys):
                            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                    ((mean(xs)**2) - mean(xs**2)))
                            return m

                        oof[e] = best_fit_slope(xs,ys)
                        
                    #wake
                    TimeF[s+1,22] = (oof[0] + oof[1]) / 2  # frontal channels
                    TimeF[s+1,23] = (oof[2] + oof[3]) / 2  # central channels
                    TimeF[s+1,24] = (oof[4] + oof[5]) / 2  # posterior channels
                    
                    
                elif isT4 > 0:
                    
                    hypno = hypno.astype('str')
                    
                    if ix_1 + ix_2 + ix_3 + ix_4 > 0:   #they slept
                        
                        #compute band power
                        where_sleep = np.isin(hypno, ['1.0','2.0','3.0','4.0'])  # True if sample is in N2 / N3, False otherwise
                        data_sleep = data[:, where_sleep]        

                        win = int(4 * SF)  # Window size is set to 4 seconds
                        freqs, psd = welch(data_sleep, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                        oof = np.zeros(len(data), dtype=np.float64)
                        for e, elec in enumerate(data):
                            xs = np.array(psd[e,:], dtype=np.float64)
                            ys = np.array(freqs, dtype=np.float64)

                            def best_fit_slope(xs,ys):
                                m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                        ((mean(xs)**2) - mean(xs**2)))
                                return m

                            oof[e] = best_fit_slope(xs,ys)

                        #sleep
                        TimeF[s+1,10] = (oof[0] + oof[1]) / 2  # frontal channels
                        TimeF[s+1,11] = (oof[2] + oof[3]) / 2  # central channels
                        TimeF[s+1,12] = (oof[4] + oof[5]) / 2  # posterior channels
                        

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    oof = np.zeros(len(data), dtype=np.float64)
                    for e, elec in enumerate(data):
                        xs = np.array(psd[e,:], dtype=np.float64)
                        ys = np.array(freqs, dtype=np.float64)

                        def best_fit_slope(xs,ys):
                            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                    ((mean(xs)**2) - mean(xs**2)))
                            return m

                        oof[e] = best_fit_slope(xs,ys)
                        
                    #wake
                    TimeF[s+1,25] = (oof[0] + oof[1]) / 2  # frontal channels
                    TimeF[s+1,26] = (oof[2] + oof[3]) / 2  # central channels
                    TimeF[s+1,27] = (oof[4] + oof[5]) / 2  # posterior channels
                    
                    
                elif isT5 > 0:
                
                    hypno = hypno.astype('str')
                    
                    if ix_1 + ix_2 + ix_3 + ix_4 > 0:   #they slept
                        
                        #compute band power
                        where_sleep = np.isin(hypno, ['1.0','2.0','3.0','4.0'])  # True if sample is in N2 / N3, False otherwise
                        data_sleep = data[:, where_sleep]        

                        win = int(4 * SF)  # Window size is set to 4 seconds
                        freqs, psd = welch(data_sleep, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                        oof = np.zeros(len(data), dtype=np.float64)
                        for e, elec in enumerate(data):
                            xs = np.array(psd[e,:], dtype=np.float64)
                            ys = np.array(freqs, dtype=np.float64)

                            def best_fit_slope(xs,ys):
                                m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                        ((mean(xs)**2) - mean(xs**2)))
                                return m

                            oof[e] = best_fit_slope(xs,ys)

                        #sleep
                        TimeF[s+1,13] = (oof[0] + oof[1]) / 2  # frontal channels
                        TimeF[s+1,14] = (oof[2] + oof[3]) / 2  # central channels
                        TimeF[s+1,15] = (oof[4] + oof[5]) / 2  # posterior channels
                        

                    #compute band power
                    where_wake = np.isin(hypno, ['0.0'])  # True if sample is in N2 / N3, False otherwise
                    data_wake = data[:, where_wake]        

                    win = int(4 * SF)  # Window size is set to 4 seconds
                    freqs, psd = welch(data_wake, SF, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

                    oof = np.zeros(len(data), dtype=np.float64)
                    for e, elec in enumerate(data):
                        xs = np.array(psd[e,:], dtype=np.float64)
                        ys = np.array(freqs, dtype=np.float64)

                        def best_fit_slope(xs,ys):
                            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                                    ((mean(xs)**2) - mean(xs**2)))
                            return m

                        oof[e] = best_fit_slope(xs,ys)
                        
                    #wake
                    TimeF[s+1,28] = (oof[0] + oof[1]) / 2  # frontal channels
                    TimeF[s+1,29] = (oof[2] + oof[3]) / 2  # central channels
                    TimeF[s+1,30] = (oof[4] + oof[5]) / 2  # posterior channels
                

MSLT_outfile = open(PATH + 'OOF_slope.csv','w')
with MSLT_outfile:
    writer = csv.writer(MSLT_outfile,delimiter=',')
    writer.writerows(TimeF)

