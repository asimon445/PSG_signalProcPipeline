### Preprocessing stages:
# 1. apply bandpass filter from 0.3 to 40 Hz
# 2. downsample (optional) from 400 to 100 Hz
# 3. rereference (to linked mastoids for now, can be changed)
# 4. import hypnogram and upsample to equal the sampling rate of the eeg data. the hypnogram vector should be the same length as the eeg
# 5. separate the eeg into REM and NREM stages based on the hypnogram. Any non-continuous portions will be separated into their own epochs
# 6. epoch the data again so that all epochs are 15 seconds. 
# 7. reject artifacts using covariance-based method that yasa provides



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

PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/raw2/'
OUTPATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/Preprocessed_2/'

PLOT = False

## IMPORTANT! The user must make sure that every patient with a PSG file also contains a corresponding hypnogram! 
## If there is anyone without one, staging will not work. Put those files in a separate folder

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
            art, zscores = yasa.art_detect(data, SF, window=5, method='covar', threshold=3)
            art.shape, zscores.shape

            print(f'{art.sum()} / {art.size} epochs rejected.')

            threshold = 3
            perc_expected_rejected = (1 - erf(threshold / np.sqrt(2))) * 100
            print(f'{perc_expected_rejected:.2f}% of all epochs are expected to be rejected.')

            # Actual
            (art.sum() / art.size) * 100
            print(f'{(art.sum() / art.size) * 100:.2f}% of all epochs were actually rejected.')


            if PLOT:
            ### Plot the artifact vector
                plt.plot(art);
                plt.yticks([0, 1], labels=['Good (0)', 'Art (1)']);

                ### Plot a histogram of z-score distributions
                sns.distplot(zscores)
                plt.title('Histogram of z-scores')
                plt.xlabel('Z-scores')
                plt.ylabel('Density')
                plt.axvline(2, color='r', label='Threshold')
                plt.axvline(-2, color='r')
                plt.legend(frameon=False);

            # The resolution of art is 5 seconds, so its sampling frequency is 1/5 (= 0.2 Hz)
            sf_art = 1 / 5
            art_up = yasa.hypno_upsample_to_data(art, sf_art, data, SF)
            art_up.shape, hypno_up.shape

            # Add -1 to hypnogram where artifacts were detected
            hypno_with_art = hypno_up.copy()
            hypno_with_art[art_up] = '-1'

            # The staging file has REM coded as '5', we need it coded as '4'
            for r in range(len(hypno_with_art)):
                if hypno_with_art[r] == '5':
                    hypno_with_art[r] = '4'

            # I used this during pipeline development to check that each subject ended up with hypnograms that reflected their actual sleep stages
            #fig = yasa.plot_spectrogram(data[0,], SF, hypno=hypno_with_art, fmax=30, cmap='Spectral_r', trimperc=5)
            
            ## Save here
            picks = mne.pick_types(eeg.info, meg=False, eeg=True, eog=False)

            eeg.save(OUTPATH + PSG_fname[np.s_[0:len(PSG_fname)-4]]+ '_preprocessed.fif', picks=picks, overwrite=True)

            hypno_outfile = open(OUTPATH + Hyp_fname[np.s_[0:len(Hyp_fname)-4]] + '_with_artifacts.csv','w')
            with hypno_outfile:
                writer = csv.writer(hypno_outfile,delimiter=' ')
                writer.writerows(hypno_with_art)






            # Proportion of each stage in ``hypno_with_art``
         #   pd.Series(hypno_with_art).value_counts(normalize=True)

            # Plot new hypnogram and spectrogram on F3
         #   yasa.plot_spectrogram(data[0, :], SF, hypno_with_art);

            ### Slow wave detection
         #   sw = yasa.sw_detect(data, SF, ch_names=channels, hypno=hypno_with_art, include=(2, 3), freq_sw=(0.5, 1.6))

            # Get the average per channel and stage
         #   sw.summary(grp_chan=True, grp_stage=True, aggfunc='mean')

            # Plot an average template of the detected slow-waves, centered around the negative peak
         #   ax = sw.plot_average(center="NegPeak", time_before=0.4, time_after=0.8, palette="Set1")
         #  ax.legend(frameon=False)
         #  sns.despine()


            #newHyp = hypnog[:].repeat(SF/(1/30), axis = 0)

            #how long is the EEG file in seconds?
            #numSecs[f,0] = len(data[0])/SF
            #numSecs[f,1] = len(newHyp)/SF

            ### make sure that eeg and hypno are the same length
            # they are very close in length (~2.5 minutes difference), but still not perfect. Ask Leslie about how to determine when hypno was started with respect to EEG and align based on that!

            #split based on NREM and REM

            #epoch




            #option 1: use an EOG defined-threshold for removing artifacts
            #if this is what we use, we'll keep the EOG. If not, then we'll remove it so that the covariance method is more likely to work

            #reject = dict(eog=150e-6)
            #epochs_params = dict(events=events, event_id=event_id, tmin=tmin, tmax=tmax,
            #                 picks=picks, reject=reject, proj=True)

            #option 2: use the covariance-based method provided by YASA 
                # - this will in threory detect ocular and other artifacts, where EOG is more likely to just detect ocular. Option 2 is prob better

            # artifact rejection
            #art, zscores = yasa.art_detect(data, SF, window=5, method='covar', threshold=3)
            #art.shape, zscores.shape

            #print(f'{art.sum()} / {art.size} epochs rejected.')

            ### Plot the artifact vector
            #plt.plot(art);
            #plt.yticks([0, 1], labels=['Good (0)', 'Art (1)']);

            ### Plot a histogram of z-score distributions
            #sns.distplot(zscores)
            #plt.title('Histogram of z-scores')
            #plt.xlabel('Z-scores')
            #plt.ylabel('Density')
            #plt.axvline(2, color='r', label='Threshold')
            #plt.axvline(-2, color='r')
            #plt.legend(frameon=False);

    else:
        
        "The PSG file: %s and the hypnogram file: %s are the same index in their lists, but are not the same person"   % (PSG_fnum, Hyp_fnum)

        
        
        
        
####### everything below here is not part of the typical pipeline -- it was for manually correcting errors that I encountered along the way
        
f = 42
        
eeg = mne.io.read_raw_fif(FILES[f], preload=True)
hypnogram = np.genfromtxt(HYPNO[f], delimiter=',',dtype=str,'formats': ('S2'))

for r in range(len(hypnogram)):
    if hypnogram[r] == '- 1':
        hypnogram[r] = '-1'
    elif hypnogram[r] == '2 1':
        hypnogram[r] = '2'
    elif hypnogram[r] == '-':
        hypnogram[r] = '-1'
        
data = eeg.get_data() 
F3 = []
F3 = data[0,:]

fig = yasa.plot_spectrogram(F3, SF, hypno=hypnogram, fmax=30, cmap='Spectral_r', trimperc=5)

Hyp_fname = HYPNO[f][np.s_[pathsep+1:len(FILES[f])]]


hypno_outfile = open(OUTPATH + Hyp_fname,'w')
with hypno_outfile:
    writer = csv.writer(hypno_outfile,delimiter=' ')
    writer.writerows(hypno_with_art)
    
    

datan = np.delete(data,np.s_[3812200:4117800],1)
F3 = []
F3 = datan[0,:]
fig = yasa.plot_spectrogram(F3, SF, hypno=hypnogram, fmax=30, cmap='Spectral_r', trimperc=5)
    
    
    

    
eeg.crop(tmin=0,tmax=38113.99,include_tmax=True)

PSG_fname = FILES[f][np.s_[pathsep+1:len(FILES[f])]]
picks = mne.pick_types(eeg.info, meg=False, eeg=True, eog=False)
eeg.save(OUTPATH + PSG_fname, picks=picks, overwrite=True)
    





    
    
    
    
    
    
    
    
    

### Artifact rejection
art, zscores = yasa.art_detect(data, SF, window=5, method='covar', threshold=2.5)
art.shape, zscores.shape

print(f'{art.sum()} / {art.size} epochs rejected.')

threshold = 2.5
perc_expected_rejected = (1 - erf(threshold / np.sqrt(2))) * 100
print(f'{perc_expected_rejected:.2f}% of all epochs are expected to be rejected.')

# Actual
(art.sum() / art.size) * 100
print(f'{(art.sum() / art.size) * 100:.2f}% of all epochs were actually rejected.')

# The resolution of art is 5 seconds, so its sampling frequency is 1/5 (= 0.2 Hz)
sf_art = 1 / 5
art_up = yasa.hypno_upsample_to_data(art, sf_art, data, SF)
art_up.shape, hypnogram.shape

# Add -1 to hypnogram where artifacts were detected
hypno_with_art = hypnogram.copy()
hypno_with_art[art_up] = -1

hypno_outfile = open(OUTPATH + Hyp_fname,'w')
with hypno_outfile:
    writer = csv.writer(hypno_outfile,delimiter=' ')
    writer.writerows(hypno_with_art)
    
eeg.save(OUTPATH + PSG_fname, picks=picks, overwrite=True)