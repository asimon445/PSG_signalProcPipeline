## this lil script will preprocess the MSLT data from the PSP/CBS/other NDD study. For this to work properly, each participant needs their own subfolder within a larger directory. Each subject's folder should contain hypnograms and PSG files from each trial. Trials in which the participant did not sleep should have 'no sleep' in the PSG file name. Trials should be indicated by 'T1' or 'T2', etc... in the file name. 

## The script will more or less follow the same preprocessing steps as the overnight PSG processing pipeline. Those steps are as follows:
# 1. apply bandpass filter from 0.3 to 50 Hz
# 2. downsample from 400 to 100 Hz
# 3. rereference (to linked mastoids for now, can be changed)
# 4. import hypnogram and upsample to equal the sampling rate of the eeg data. 
# 5. the length of the hypnogram is matched to the length of the PSG data. 
#  ******* IMPORTANT ********  Step #5 often does not work and will at some point require the user to visually inspect the data to be sure this was done properly
# 6. label artifacts using 'yasa.art_detect' and selecting the "covariance-based" parameter (portions of the hypnogram are labeled '-1' to denote a segment that contains artifact)
#    THRESHOLD IS SET TO 3 ZSCORES FOR ART DETECTION -- THIS CAN BE CHANGED ON LINES XXX AND XXX (CHANGE IT ON BOTH IF YOU"RE GOING TO CHANGE IT AT ALL!!!!!!!!!!!!!!!!)
# 
# NOTE: PSG data are NOT epoched -- they are kept continuous 

# IF there is no sleep, the hypnogram will not be imported (was not exported from Prana in this study, so nothing to import). However, we will need a hypnogram for artifact rejection. So if there's no sleep in a trial, we'll make our own hypnogram that is the same length as the PSG data (it will be all 0's at first), and then if there is an artifact in a segment of data, we will label that in the newly created hypnogram as '-1'.


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


PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/MSLT/raw/'
OUTPATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/MSLT/preprocessed_1sec_windows_1point5_2/'

SUBDIRS = glob.glob(PATH + '/*/')   #list subdirectories
SUBDIRS.sort()

SF = 100    #Define frequency to downsample to -- Nyquist rate is 90 Hz for resolving spectral power in frequencies up to 45 Hz (gamma = 30-45 Hz -- per Walsh et al., 2017) 
            #VERY IMPORTANT THAT THE SAMPLING RATE MUST BE KEPT ABOVE THIS
            #Raw data is sampled at 400 Hz -- we don't need to downsample but can if we want to reduce size of data

failures = []   #stores list of edfs without a hypnogram on trials with sleep
artifact = []
subs = []

thresh = 1.5   #this is the z-score value that we are using to detect artifacts. Default is 3, but after visually inspecting preprocessed data, it can/should be changed for files that need a different threshold. Just be sure to delete those files from the preprocessed folder before trying to run this again with a different threshold, otherwise they'll be skipped!

for s in range(len(SUBDIRS)):

    # this is for indexing the last '/' in the path so that we can pull the filename easily for each participant
    sep = '/'
    currdir = SUBDIRS[s]
    def find(currdir, sep):
        return [i for i, ltr in enumerate(currdir) if ltr == sep]

    pathseps = list(find(currdir,sep))
    pathsep = pathseps[len(pathseps)-1]   #this here is the position of the final '/'
    
    # generate lists of .edf and .xls files
    edfs = glob.glob(currdir + '*.edf')
    edfs.sort()
    
    xlss = glob.glob(currdir + '*.csv')
    xlss.sort()

    for f in range(len(edfs)):
        
        nosleep = 'no sleep'

        nsleep = edfs[f].lower().find(nosleep)

        if nsleep < 0:    #if this person slept during this MSLT trial, then load in the corresponding hypnogram

            #check if there is a hypnogram associated with this PT. There were a couple of subjects who had missing hypnograms for a trial that they fell asleep on

            isT1 = edfs[f].find('T1')
            isT2 = edfs[f].find('T2')
            isT3 = edfs[f].find('T3')
            isT4 = edfs[f].find('T4')
            isT5 = edfs[f].find('T5')

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
                PSG_fname = edfs[f][np.s_[pathsep+1:len(edfs[0])]]
                
                # check if the file has already been preprocessed. If it has, skip it. 
                if not os.path.isfile(OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_preprocessed.fif'):

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

                    #load hypnogram for this same subject
                    hypnogram = []
                    hypnogram = np.loadtxt(fname = xlss[xl_match],dtype = 'str',delimiter = ',',skiprows=0)  

                    #transform hypnogram file into single column vector of stage info
                    hypnog = []
                    hypnog = hypnogram[:,1]
        
                    empt = 0
                    for r in range(len(hypnog)):
                        if empt == 0:
                            if hypnog[r] == '':
                                hypnog = np.delete(hypnog,np.s_[r:len(hypnog)],0)
                                empt = 1

                    #find any portions of the hypnogram where it went offline and fill in the gaps with "-1" (artifact)
                    for h in range(len(hypnog)-1):

                        thisIter = hypnogram[h,0]
                        nextIter = hypnogram[h+1,0]

                        if len(nextIter) > 0:

                            thisIter = datetime.datetime.strptime(thisIter,"%H:%M:%S")
                            nextIter = datetime.datetime.strptime(nextIter,"%H:%M:%S")

                            #check if the difference between two rows of hypnogram timestamps are greater than 30 seconds. If they are, that means that it went offline and we need to adjust the length of the hypnogram to account for the time it went offline
                            IterDiff = nextIter - thisIter
                            IterDiff = IterDiff.total_seconds()

                            if IterDiff > 30:

                                NumGaps = int(IterDiff/30)

                                for g in range(NumGaps):

                                    hypnog = np.insert(hypnog,[h+g+1,0],[-1])

                    #align time stamp for start of EEG with hypnogram
                    startEEG = eeg.info["meas_date"]
                    start_dtEEG = startEEG.strftime("%H:%M:%S")

                    timeEEG = datetime.datetime.strptime(start_dtEEG,"%H:%M:%S")

                    start_dtHyp = hypnogram[0,0]   #first value in timestamp column
                    timeHYP = datetime.datetime.strptime(start_dtHyp,"%H:%M:%S")

                    diff = timeHYP - timeEEG

                    SecDiff = diff.total_seconds()    #this is the difference between start time for EEG and Hypnogram in seconds

                    if SecDiff > 0:        #If EEG was started earlier than hypnogram

                        #remove the first (diff * sfreq) time points from EEG data
                        nIter = int(SecDiff*sfreq)

                        data = np.delete(data,np.s_[0:nIter],1)

                    elif SecDiff < 0:     #If hypnogram was started earlier than EEG

                        #remove the first (diff * sfreq) time points from hypnogram 
                        nIter = int(-SecDiff/30)

                        hypnog = np.delete(hypnog,np.s_[0:nIter],0)
                
                    #upsample the hypnogram to have the same sampling freq as the EEG data   
                    hypno_up = yasa.hypno_upsample_to_data(hypno=hypnog, sf_hypno=(1/30), data=data, sf_data=SF)
                    print(hypno_up.size == data.shape[1])  # Does the hypnogram have the same number of samples as data?
                    print(hypno_up.size, 'samples:', hypno_up)

                    ### Artifact rejection
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
                    sf_art = 1 / 5
                    art_up = yasa.hypno_upsample_to_data(art, sf_art, data, SF)
                    art_up.shape, hypno_up.shape

                    # Add -1 to hypnogram where artifacts were detected                  
                    hypno_with_art = hypno_up.copy()
                    hypno_with_art = hypno_with_art.astype('U8')  
                    hypno_with_art[art_up] = '-1'

                    # The staging file has REM coded as '5', we need it coded as '4'
                    for r in range(len(hypno_with_art)):
                        if hypno_with_art[r] == '5':
                            hypno_with_art[r] = '4'

                    ## Save here
                    outdir = OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/'
                    
                    if not os.path.isdir(outdir):
                        os.mkdir(outdir) 
                    
                    picks = mne.pick_types(eeg.info, meg=False, eeg=True, eog=False)

                    eeg.save(OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_preprocessed.fif', picks=picks, overwrite=True)

                    hypno_outfile = open(OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/' + Hyp_fname[np.s_[0:len(Hyp_fname)-4]] + '_with_artifacts.csv','w')
                    with hypno_outfile:
                        writer = csv.writer(hypno_outfile,delimiter=' ')
                        writer.writerows(hypno_with_art)
                            
            else:

                failures.append(edfs[f])   #list files with sleep but without a hypnogram here. Save at end

        elif nsleep > 0:     #if they did not sleep, then just go ahead and load/process the PSG data and treat it as resting wake
            
            subpathseps = list(find(SUBDIRS[s],sep))
            subpathsep = subpathseps[len(pathseps)-1]   #this here is the position of the final '/'
                
            PSG_fname = edfs[f][np.s_[pathsep+1:len(edfs[0])]]
            
            # check if the file has already been preprocessed. If it has, skip it. 
            if not os.path.isfile(OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_preprocessed.fif'):

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
                sf_art = 1 / 5
                art_up = yasa.hypno_upsample_to_data(art, sf_art, data, SF)
                art_up.shape, hypnogram.shape

                # Add -1 to hypnogram where artifacts were detected                  
                hypno_with_art = hypnogram.copy()
                hypno_with_art = hypno_with_art.astype('U8')  
                hypno_with_art[art_up] = '-1'

                ## Save here
                outdir = OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/'
                    
                if not os.path.isdir(outdir):
                    os.mkdir(outdir) 
                    
                picks = mne.pick_types(eeg.info, meg=False, eeg=True, eog=False)

                eeg.save(OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_preprocessed.fif', picks=picks, overwrite=True)

                hypno_outfile = open(OUTPATH + SUBDIRS[s][subpathseps[len(subpathseps)-2]:subpathseps[len(subpathseps)-1]] + '/' + PSG_fname[np.s_[0:len(PSG_fname)-4]] + '_SCR20_with_artifacts.csv','w')
                with hypno_outfile:
                    writer = csv.writer(hypno_outfile,delimiter=' ')
                    writer.writerows(hypno_with_art)


#save artifact info
art_output = np.vstack((subs,artifact))

art_outfile = open(OUTPATH + 'artifact_rejection_info_2_point_5.csv','w')
with art_outfile:
    writer = csv.writer(art_outfile,delimiter=' ')
    writer.writerows(art_output)


#save failures
fails_outfile = open(OUTPATH + 'failures_info_2_point_5.csv','w')
with fails_outfile:
    writer = csv.writer(fails_outfile,delimiter=' ')
    writer.writerows(failures)
