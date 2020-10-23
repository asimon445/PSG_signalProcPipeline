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
from statistics import mean

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

#format output header  -- one over f slope
Stats = np.zeros((len(FILES)+1,13),dtype=np.int)
Stats = Stats.astype('U20')

Stats[0,0] = 'Subject'
Stats[0,1] = 'NREM F3 oof'
Stats[0,2] = 'NREM F4 oof'
Stats[0,3] = 'NREM C3 oof'
Stats[0,4] = 'NREM C4 oof'
Stats[0,5] = 'NREM O1 oof'
Stats[0,6] = 'NREM O2 oof'
Stats[0,7] = 'REM F3 oof'
Stats[0,8] = 'REM F4 oof'
Stats[0,9] = 'REM C3 oof'
Stats[0,10] = 'REM C4 oof'
Stats[0,11] = 'REM O1 oof'
Stats[0,12] = 'REM O2 oof'




PSPspec_front = np.zeros((22,201),dtype=np.str)
PSPspec_front = PSPspec_front.astype('U20')

PSPspec_central = np.zeros((22,201),dtype=np.str)
PSPspec_central = PSPspec_central.astype('U20')

PSPspec_post = np.zeros((22,201),dtype=np.str)
PSPspec_post = PSPspec_post.astype('U20')

PSP_remspec_front = np.zeros((22,201),dtype=np.str)
PSP_remspec_front = PSP_remspec_front.astype('U20')

PSP_remspec_central = np.zeros((22,201),dtype=np.str)
PSP_remspec_central = PSP_remspec_central.astype('U20')

PSP_remspec_post = np.zeros((22,201),dtype=np.str)
PSP_remspec_post = PSP_remspec_post.astype('U20')


Conspec_front = np.zeros((25,201),dtype=np.str)
Conspec_front = Conspec_front.astype('U20')

Conspec_central = np.zeros((25,201),dtype=np.str)
Conspec_central = Conspec_central.astype('U20')

Conspec_post = np.zeros((25,201),dtype=np.str)
Conspec_post = Conspec_post.astype('U20')

Con_remspec_front = np.zeros((25,201),dtype=np.str)
Con_remspec_front = Con_remspec_front.astype('U20')

Con_remspec_central = np.zeros((25,201),dtype=np.str)
Con_remspec_central = Con_remspec_central.astype('U20')

Con_remspec_post = np.zeros((25,201),dtype=np.str)
Con_remspec_post = Con_remspec_post.astype('U20')

psp = 0
con = 0

psp_rem = 0
con_rem = 0

diagnosis = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,1,1,1,1,0,0,2,0,2,2,0,0,0,0,0,2,2,2,1,1,0,2,2,0,0,2,2,2,2,0,2,0,0,2,2,0,2,2,2,1,1,2,2]


for f, file in enumerate(FILES):
            
    if (f != 47) & (f != 60) & (f != 62):
        
        PSG_fnum = FILES[f][np.s_[pathsep+1:pathsep+6]]
        
        Stats[f+1,0] = PSG_fnum
        
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

        where_NREM = np.isin(hypnogram, ['2', '3'])  # True if sample is in N2 / N3, False otherwise
        data_NREM = data[:, where_NREM] 

        win = int(4 * sf)  # Window size is set to 4 seconds
        freqs, psd = welch(data_NREM, sf, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz

        for e, elec in enumerate(data):
            xs = np.array(psd[e,:], dtype=np.float64)
            ys = np.array(freqs, dtype=np.float64)
            
            def best_fit_slope(xs,ys):
                m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                     ((mean(xs)**2) - mean(xs**2)))
                return m

            Stats[f+1,e+1] = best_fit_slope(xs,ys)
        

        if diagnosis[f] == 1:
            Conspec_front[con] = (psd[0,:] + psd[1,:])/2
            Conspec_central[con] = (psd[2,:] + psd[3,:])/2
            Conspec_post[con] = (psd[4,:] + psd[5,:])/2
            con = con+1
            
        elif diagnosis[f] == 2:
            PSPspec_front[psp] = (psd[0,:] + psd[1,:])/2
            PSPspec_central[psp] = (psd[2,:] + psd[3,:])/2
            PSPspec_post[psp] = (psd[4,:] + psd[5,:])/2
            psp = psp+1
                
        where_REM = np.isin(hypnogram, ['4'])  # True if sample is in N2 / N3, False otherwise
        data_REM = data[:, where_REM] 
        
        win = int(4 * sf)  # Window size is set to 4 seconds
        freqs_rem, psd_rem = welch(data_REM, sf, nperseg=win, average='median')  # Works with single or multi-channel data, freq bins are 0.25 Hz
        
        
        for e, elec in enumerate(data):
            xs = np.array(psd[e,:], dtype=np.float64)
            ys = np.array(freqs, dtype=np.float64)
            
            def best_fit_slope(xs,ys):
                m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                     ((mean(xs)**2) - mean(xs**2)))
                return m

            Stats[f+1,e+7] = best_fit_slope(xs,ys)
        
        if diagnosis[f] == 1:
            Con_remspec_front[con_rem] = (psd[0,:] + psd[1,:])/2
            Con_remspec_central[con_rem] = (psd[2,:] + psd[3,:])/2
            Con_remspec_post[con_rem] = (psd[4,:] + psd[5,:])/2
            con_rem = con_rem+1
            
        elif diagnosis[f] == 2:
            PSP_remspec_front[psp] = (psd[0,:] + psd[1,:])/2
            PSP_remspec_central[psp] = (psd[2,:] + psd[3,:])/2
            PSP_remspec_post[psp] = (psd[4,:] + psd[5,:])/2
            psp_rem = psp_rem+1
            
            
        
SpecPow_outfile = open('/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/' + '1 over f slope.csv','w')
with SpecPow_outfile:
    writer = csv.writer(SpecPow_outfile,delimiter=',')
    writer.writerows(Stats)
    
    
    
    
    
PSP_NREM_1f_front = PSPspec_front.astype(np.float)
PSP_NREM_1f_front = np.average(PSP_NREM_1f_front,axis=0)

PSP_NREM_1f_central = PSPspec_central.astype(np.float)
PSP_NREM_1f_central = np.average(PSP_NREM_1f_central,axis=0)

PSP_NREM_1f_post = PSPspec_post.astype(np.float)
PSP_NREM_1f_post = np.average(PSP_NREM_1f_post,axis=0)

Con_NREM_1f_front = Conspec_front.astype(np.float)
Con_NREM_1f_front = np.average(Con_NREM_1f_front,axis=0)

Con_NREM_1f_central = Conspec_central.astype(np.float)
Con_NREM_1f_central = np.average(Con_NREM_1f_central,axis=0)

Con_NREM_1f_post = Conspec_post.astype(np.float)
Con_NREM_1f_post = np.average(Con_NREM_1f_post,axis=0)


#PSPvar = PSP1.std(axis=0)
#Convar = Con.std(axis=0)


    # Plot
plt.plot(freqs, psg[0,:], 'r', lw=2)
plt.fill_between(freqs, PSP_avg, cmap='Spectral',facecolor='red', alpha=0.5)
plt.plot(freqs,Con_avg,'b', lw=2)
plt.fill_between(freqs, Con_avg, cmap='Spectral',facecolor='blue', alpha=0.5)
plt.xlim(0, 50)
plt.yscale('log')
sns.despine()
plt.title('Frontal Power Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD log($uV^2$/Hz)');


