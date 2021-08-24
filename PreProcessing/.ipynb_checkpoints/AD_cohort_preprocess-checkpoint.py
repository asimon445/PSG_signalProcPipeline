## This is a slightly modified version of the original PSG preprocessing pipeline that was created for the PSP cohort. 
## This one is meant to process data from the AD cohort. The only difference is this has a different electrode montage (more of them).

## IMPORTANT! The user must make sure that every patient with a PSG file also contains a corresponding hypnogram! 
## Each PSG file should have a corresponding hypnogram for every step of processing and analysis from here on out!

### Preprocessing stages:
# 1. apply bandpass filter from 0.3 to 50 Hz
# 2. downsample from 400 to 100 Hz
# 3. rereference (to linked mastoids for now, can be changed)
# 4. import hypnogram and upsample to equal the sampling rate of the eeg data. 
# 5. the length of the hypnogram is matched to the length of the PSG data. 
#  ******* IMPORTANT ********  Step #5 often does not work and will at some point require the user to visually inspect the data to be sure this was done properly
# 6. OPTIONAL (recommended for limiting to overnight analysis) -- trim data from 9pm to 9am (to exclude confounds of potential naps during day)
# 7. label artifacts using 'yasa.art_detect' and selecting the "covariance-based" parameter (portions of the hypnogram are labeled '-1' to denote a segment that contains artifact)
#    THRESHOLD IS SET TO 3 ZSCORES FOR ART DETECTION -- THIS CAN BE CHANGED ON LINES 182 AND 187 (CHANGE IT ON BOTH IF YOU"RE GOING TO CHANGE IT AT ALL!!!!!!!!!!!!!!!!)
# 
# NOTE: PSG data are NOT epoched -- they are kept continuous 



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

COHORT = ['retry'] #['Amnestic','Control','lvPPA','PCA']

for co in range(len(COHORT)):
    PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/AD_cohort/raw_forFewerElecs/%s/' % (COHORT[co])
    OUTPATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/AD_cohort/Preprocessed_fewerElecs/%s/' % (COHORT[co])

    thresh = 2.5   #artifact rejection threshold (stdev -- usually 1.5 to 2 is sufficient, but should be reviewed on a subject to subject basis)
    TRIM = True   #set to 'True' if you want to trim the data from 9pm to 9am (recommended for limiting analysis to overnight data)

    artifact = []
    subs = []
    failures = []

    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)

    # There's some code that I used to plot artifact rejection info here -- it's pretty useful when first using the pipeline and first
    # trying to figure out the parameters to use for artifact detection thresholds
    PLOT = False

    #create list of PSG files
    FILES = glob.glob(PATH + '*.edf')
    FILES.sort()

    #create list of hypnogram files  
    HYPNO = glob.glob(PATH + '*.csv')
    HYPNO.sort()

    SF = 100    #Define frequency to downsample to -- Nyquist rate is 90 Hz for resolving spectral power in frequencies up to 45 Hz (gamma = 30-45 Hz -- per Walsh et al., 2017) 
                #VERY IMPORTANT THAT THE SAMPLING RATE MUST BE KEPT ABOVE THIS
                #Raw data is sampled at 400 Hz -- we don't need to downsample but can if we want to reduce size of data

    numSecs = np.zeros((len(FILES),2), dtype=np.int)

    # this is for indexing the last '/' in the path so that we can pull the filename easily for each participant
    sep = '/'
    def find(PATH, sep):
        return [i for i, ltr in enumerate(PATH) if ltr == sep]

    pathsep = list(find(PATH,sep))
    pathsep = pathsep[len(pathsep)-1]


    for f, file in enumerate(FILES):

        ### be sure to check that FILES[f] is the same sub as HYPNO[f] !!!
        PSG_fnum = FILES[f][np.s_[pathsep+1:pathsep+6]]
        Hyp_fnum= HYPNO[f][np.s_[pathsep+1:pathsep+6]]

        Hyp_fname = HYPNO[f][np.s_[pathsep+1:len(FILES[f])]]
        PSG_fname = FILES[f][np.s_[pathsep+1:len(FILES[f])]]

        # make sure that the participant numbers between the PSG file and the hypnogram file for iteration 'f' are the same
        if PSG_fnum == Hyp_fnum:

            # check if the file has already been preprocessed. If it has, skip it. 
            if not os.path.isfile(OUTPATH + PSG_fname[np.s_[0:len(PSG_fname)-4]]+ '_preprocessed.fif'):

                #load PSG file
                eeg = mne.io.read_raw_edf(FILES[f], preload=True)

                # Apply a bandpass filter from 0.3 to 50 Hz 
                eeg.filter(0.3, 50)             

                #Downsample to SF 
                eeg.resample(SF)   

                #select the channels where eeg was recorded, discard the rest
#                 eeg.pick_channels(['EEG Fp1-Ref','EEG F7-Ref','EEG T7-Ref','EEG P7-Ref','EEG O1-Ref','EEG F3-Ref','EEG C3-Ref','EEG P3-Ref','EEG Fz-Ref','EEG Cz-Ref','EEG Fp2-Ref','EEG F8-Ref','EEG T8-Ref','EEG P8-Ref','EEG O2-Ref','EEG F4-Ref','EEG C4-Ref','EEG P4-Ref','EEG Fpz-Ref','EEG Pz-Ref','EEG TP7-Ref','EEG CP5-Ref','EEG CP3-Ref','EEG P9-Ref','EEG P5-Ref','EEG P09-Ref','EEG P07-Ref'])  # Select a subset of EEG channels
                
                eeg.pick_channels(['EEG O1-Ref','EEG F3-Ref','EEG C3-Ref','EEG O2-Ref','EEG F4-Ref','EEG C4-Ref'])  # Select a subset of EEG channels
                
                #re-reference EEG to linked-mastoids, as opposed to the contralateral mastoid reference that the raw signal is referenced to
                eeg.set_eeg_reference('average', projection=True)

                #store EEG data into an e x t matrix, where e = num elecs and t = samples
                data = eeg.get_data() 

                #convert data from Volts to µV
                data = data*1000000

                info = eeg.info
                channels = eeg.ch_names
                sfreq = eeg.info["sfreq"]

                #load hypnogram for this same subject
                hypnogram = []
                hypnogram = np.loadtxt(fname = HYPNO[f],dtype = 'str',delimiter = ',',skiprows=1)  

                #transform hypnogram file into single column vector of stage info
                hypnog = []
                hypnog = hypnogram[:,3]

                empt = 0
                for r in range(len(hypnog)):
                    if empt == 0:
                        if hypnog[r] == '':
                            hypnog = np.delete(hypnog,np.s_[r:len(hypnog)],0)
                            empt = 1

                #find any portions of the hypnogram where it went offline and fill in the gaps with "-1" (artifact)
                for h in range(len(hypnog)-1):

                    thisIter = hypnogram[h,2]
                    nextIter = hypnogram[h+1,2]

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

                start_dtHyp = hypnogram[0,2]   #first value in timestamp column
                timeHYP = datetime.datetime.strptime(start_dtHyp,"%H:%M:%S")

                diff = timeHYP - timeEEG

                SecDiff = diff.total_seconds()    #this is the difference between start time for EEG and Hypnogram in seconds
                newtimeEEG = []

                try:
                    if SecDiff > 0:        #If EEG was started earlier than hypnogram

                        #remove the first (diff * sfreq) time points from EEG data
                        nIter = int(SecDiff*sfreq)

                        xmax = data.shape[1]
                        data = np.delete(data,np.s_[0:nIter],1)
                        eeg.crop(tmin=float(nIter/100),tmax=float((xmax/100)-.0100),include_tmax=True)  

                        # update the start of the EEG recording to reflect the trimmed data
                        time_change = datetime.timedelta(seconds=int(SecDiff))
                        newtimeEEG = timeEEG + time_change

                    elif SecDiff < 0:     #If hypnogram was started earlier than EEG

                        #remove the first (diff * sfreq) time points from hypnogram 
                        nIter = int(-SecDiff/30)

                        hypnog = np.delete(hypnog,np.s_[0:nIter],0)

                    if newtimeEEG == []:
                        newtimeEEG = timeEEG   #set newtimeEEG = timeEEG so that I only use newtimeEEG from here on out

                    #upsample the hypnogram to have the same sampling freq as the EEG data   
                    hypno_up = yasa.hypno_upsample_to_data(hypno=hypnog, sf_hypno=(1/30), data=data, sf_data=SF)
                    print(hypno_up.size == data.shape[1])  # Does the hypnogram have the same number of samples as data?
                    print(hypno_up.size, 'samples:', hypno_up)
                    xmax = data.shape[1]

                    #remove any data before 9pm and after 9am
                    if TRIM == True:

                        newStart = '21:00:00'
                        newStartEEG = datetime.datetime.strptime(newStart,"%H:%M:%S")
                        Startdiff = newStartEEG - newtimeEEG
                        SecStartDiff = Startdiff.total_seconds()

                        start_ix = 0
                        end_ix = []   #if end_ix is not empty, then we'll crop out anything afterward after removing front end of data

                        if SecStartDiff > 0:        #If EEG was started before 9pm

                            #remove the first (SecStartDiff * sfreq) time points from EEG data
                            nIter = int(SecStartDiff*sfreq)

                            data = np.delete(data,np.s_[0:nIter],1)
                            eeg.crop(tmin=float(nIter/100),tmax=float((xmax/100)-.0100),include_tmax=True)   

                            hypno_up = np.delete(hypno_up,np.s_[0:nIter],0)

                            start_ix = nIter+1

                            if data.shape[1] > 4320000:
                                # compute the number of samples after 9pm that 9am occurs 
                                Enddiff = 12*60*60*100  # 12 hours * 60 min * 60 sec * 100 samples
                                datalen = data.shape[1]
                                hypnolen = len(hypno_up)

                                # if the recording is longer than 12 hours, remove anything after the 12 hr mark
                                if datalen > Enddiff:

                                    data = np.delete(data,np.s_[Enddiff:datalen],1)
                                    hypno_up = np.delete(hypno_up,np.s_[Enddiff:hypnolen],0)
                                    end_ix = datalen - (datalen - Enddiff)
                                    eeg.crop(tmin=0,tmax=float((end_ix/100)-.0100),include_tmax=True) 

                        elif SecStartDiff < 0:        #If EEG was started after 9pm

                            inverse = SecStartDiff * -1

                            nineam = 4320000 - (inverse * 100)

                            nineam = int(nineam)

                            if data.shape[1] > nineam:   #if data was still recording after 9am, cut off end of it (after 9am)

                                Enddiff = nineam  # 12 hours (in samples) minus the amount of time after 9pm that EEG was started
                                datalen = data.shape[1]
                                hypnolen = len(hypno_up)

                                # if the recording is longer than 12 hours, remove anything after the 12 hr mark
                                if datalen > Enddiff:

                                    data = np.delete(data,np.s_[Enddiff:datalen],1)
                                    hypno_up = np.delete(hypno_up,np.s_[Enddiff:hypnolen],0)
                                    end_ix = datalen - (datalen - Enddiff)
                                    eeg.crop(tmin=0,tmax=float((end_ix/100)-.0100),include_tmax=True)  

                    ### Artifact rejection
                    art, zscores = yasa.art_detect(data, SF, window=5, method='covar', threshold=thresh)
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
                    picks = mne.pick_types(eeg.info, meg=False, eeg=True, eog=False)

                    eeg.save(OUTPATH + PSG_fname[np.s_[0:len(PSG_fname)-4]]+ '_preprocessed.fif', picks=picks, overwrite=True)

                    hypno_outfile = open(OUTPATH + Hyp_fname[np.s_[0:len(Hyp_fname)-4]] + '_with_artifacts.csv','w')
                    with hypno_outfile:
                        writer = csv.writer(hypno_outfile,delimiter=' ')
                        writer.writerows(hypno_with_art)
                        
                except:
                    
                    failures.append(FILES[f])
                
        else:

            "The PSG file: %s and the hypnogram file: %s are the same index in their lists, but are not the same person"   % (PSG_fnum, Hyp_fnum)



    #save artifact info
    art_output = np.vstack((subs,artifact))

    art_outfile = open(OUTPATH + 'artifact_rejection_info_2p5.csv','w')
    with art_outfile:
        writer = csv.writer(art_outfile,delimiter=' ')
        writer.writerows(art_output)
        
    #save failures
    fails_outfile = open(OUTPATH + 'failures_info.csv','w')
    with fails_outfile:
        writer = csv.writer(fails_outfile,delimiter=' ')
        writer.writerows(failures)

