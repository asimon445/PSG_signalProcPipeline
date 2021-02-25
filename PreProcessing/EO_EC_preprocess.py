## this lil script will preprocess the MSLT/MWT data (just a few short segments from each though for RS analysis) from the PSP/CBS/other NDD study. For this to work properly, each participant needs their own subfolder within a larger directory. Each subject's folder should contain PSG files from each trial. Trials in which the participant did not sleep should have 'no sleep' in the PSG file name. Trials should be indicated by 'T1' or 'T2', etc... in the file name. 

## The script will more or less follow the same preprocessing steps as the overnight PSG processing pipeline. Those steps are as follows:
# 1. apply bandpass filter from 0.3 to 50 Hz
# 2. downsample from 400 to 100 Hz
# 3. rereference (to linked mastoids for now, can be changed)
# 4. create dummy hypnogram equal in length to the eeg data. 
# 5. label artifacts using 'yasa.art_detect' and selecting the "covariance-based" parameter (portions of the hypnogram are labeled '-1' to denote a segment that contains artifact)
#    THRESHOLD IS SET TO 3 ZSCORES FOR ART DETECTION -- THIS CAN BE CHANGED ON LINES XXX AND XXX (CHANGE IT ON BOTH IF YOU"RE GOING TO CHANGE IT AT ALL!!!!!!!!!!!!!!!!)
# 
# NOTE: PSG data are NOT epoched -- they are kept continuous 


### Author: AJ Simon ###

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


PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/Resting/raw/90 Second EO preprocess 1 stdev/'
OUTPATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/Resting/preprocessed/EO_1p0stdev_reprocessed2/'

SUBDIRS = glob.glob(PATH + '/*/')   #list subdirectories
SUBDIRS.sort()

SF = 100    #Define frequency to downsample to -- Nyquist rate is 90 Hz for resolving spectral power in frequencies up to 45 Hz (gamma = 30-45 Hz -- per Walsh et al., 2017) 
            #VERY IMPORTANT THAT THE SAMPLING RATE MUST BE KEPT ABOVE THIS
            #Raw data is sampled at 400 Hz -- we don't need to downsample but can if we want to reduce size of data

failures = []   #stores list of edfs without a hypnogram on trials with sleep
artifact = []
subs = []

thresh = 1.0   #this is the z-score value that we are using to detect artifacts. 

#for s in range(len(SUBDIRS)):
#
#   # this is for indexing the last '/' in the path so that we can pull the filename easily for each participant
sep = '/'
currdir = PATH #SUBDIRS[s]
def find(currdir, sep):
    return [i for i, ltr in enumerate(currdir) if ltr == sep]

pathseps = list(find(currdir,sep))
pathsep = pathseps[len(pathseps)-1]   #this here is the position of the final '/'
    
    # generate lists of .edf and .xls files
#    edfs = glob.glob(currdir + '*.edf')
edfs = glob.glob(PATH + '*.edf')
edfs.sort()
    
xlss = glob.glob(currdir + '*.csv')
xlss.sort()

for f in range(len(edfs)):
        
 #      subpathseps = list(find(SUBDIRS[s],sep))
 #       subpathsep = subpathseps[len(pathseps)-1]   #this here is the position of the final '/'
                
    PSG_fname = edfs[f][np.s_[pathsep+1:len(edfs[0])]]
            
        # check if the file has already been preprocessed. If it has, skip it. 
    #if not os.path.isfile(OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_preprocessed.fif'):
    if not os.path.isfile(OUTPATH + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_preprocessed.fif'):
    
        try:
                #load PSG file
            eeg = mne.io.read_raw_edf(edfs[f], preload=True)

                # Apply a bandpass filter from 0.3 to 50 Hz 
            eeg.filter(0.3, 50)             

                #Downsample to SF 
            eeg.resample(SF)   

            try:
                    #re-reference EEG to linked-mastoids, as opposed to the contralateral mastoid reference that the raw signal is referenced to
                eeg.set_eeg_reference(['M1-REF', 'M2-REF'])

                    #select the channels where eeg was recorded, discard the rest
                eeg.pick_channels(['F3-REF','F4-REF','C3-REF','C4-REF','O1-REF','O2-REF'])  # Select a subset of EEG channels
            except:
                eeg.set_eeg_reference(['M1', 'M2'])
                eeg.pick_channels(['F3','F4','C3','C4','O1','O2'])  # Select a subset of EEG channels

                #store EEG data into an e x t matrix, where e = num elecs and t = samples
            data = eeg.get_data() 

                #convert data from Volts to µV
            data = data*1000000

            info = eeg.info
            channels = eeg.ch_names
            sfreq = eeg.info["sfreq"]

                #create a hypnogram of zeros that is the same length as 'data'
            hypnogram = np.zeros(data.shape[1])

            art, zscores = yasa.art_detect(data, SF, window=1, method='covar', threshold=thresh)   #typically use 5, but for wake data 1 should be used because blinks occur frequently (e.g., every 5 seconds)
            art.shape, zscores.shape

            print(f'{art.sum()} / {art.size} epochs rejected.')

            perc_expected_rejected = (1 - erf(thresh / np.sqrt(2))) * 100
            print(f'{perc_expected_rejected:.2f}% of all epochs are expected to be rejected.')

                # Actual
            (art.sum() / art.size) * 100
            print(f'{(art.sum() / art.size) * 100:.2f}% of all epochs were actually rejected.')

            perc_rej = (art.sum() / art.size) * 100
            subs.append(PSG_fname)
            artifact.append(perc_rej)   

                # The resolution of art is 5 seconds, so its sampling frequency is 1/5 (= 0.2 Hz)
            sf_art = 1 / 1
            art_up = yasa.hypno_upsample_to_data(art, sf_art, data, SF)
            art_up.shape, hypnogram.shape

                # Add -1 to hypnogram where artifacts were detected                  
            hypno_with_art = hypnogram.copy()
            hypno_with_art = hypno_with_art.astype('U8')  
            hypno_with_art[art_up] = '-1'

                ## Save here
            outdir = OUTPATH + '/'     # OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/'

            if not os.path.isdir(outdir):
                os.mkdir(outdir) 

            picks = mne.pick_types(eeg.info, meg=False, eeg=True, eog=False)

            #eeg.save(OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_preprocessed.fif', picks=picks, overwrite=True)
            eeg.save(OUTPATH + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_preprocessed.fif', picks=picks, overwrite=True)

            #hypno_outfile = open(OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_SCR20_with_artifacts.csv','w')
            hypno_outfile = open(OUTPATH + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_SCR20_with_artifacts.csv','w')
            
            with hypno_outfile:
                writer = csv.writer(hypno_outfile,delimiter=' ')
                writer.writerows(hypno_with_art)
                    
        except:
            failures.append(edfs[f]) 


#save artifact info
art_output = np.vstack((subs,artifact))

art_outfile = open(OUTPATH + 'artifact_rejection_info_1_point_0.csv','w')
with art_outfile:
    writer = csv.writer(art_outfile,delimiter=' ')
    writer.writerows(art_output)


#save failures
fails_outfile = open(OUTPATH + 'failures_info_1_point_0.csv','w')
with fails_outfile:
    writer = csv.writer(fails_outfile,delimiter=' ')
    writer.writerows(failures)
