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

PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/'

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

for f, file in enumerate(FILES):
    
    #load PSG file
    eeg = mne.io.read_raw_edf(FILES[f], preload=True)
    
    # Apply a bandpass filter from 0.3 to 40 Hz 
    eeg.filter(0.3, 40)             
    
    #Downsample to SF 
    eeg.resample(SF)   
    
    picks = mne.pick_types(eeg.info, meg=False, eeg=True, eog=True)
    
    #re-reference EEG to linked-mastoids, as opposed to the contralateral mastoid reference that the raw signal is referenced to
    eeg.set_eeg_reference(['M1-REF', 'M2-REF'])
    
    #load hypnogram for this same subject
    hypnog = np.loadtxt(fname = HYPNO[f],dtype = 'str',delimiter = ',',skiprows=5)  
    
    ### be sure to check that FILES[f] is the same sub as HYPNO[f] !!!
    
    #transform hypnogram file into single column vector of stage info
    hypnog = hypnog[:,3]
    
    #upsample the hypnogram to have the same sampling freq as the EEG data 
    newHyp = hypnog[:].repeat(SF/(1/30), axis = 0)
    
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
        
    # Select a subset of EEG channels
    eeg.pick_channels(['F3-REF','F4-REF','C3-REF','C4-REF','O1-REF','O2-REF'])  
    data = eeg.get_data()     
    
    # artifact rejection
    art, zscores = yasa.art_detect(data, SF, window=5, method='covar', threshold=3)
    art.shape, zscores.shape
    
    print(f'{art.sum()} / {art.size} epochs rejected.')
    
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

    
    
    eeg.pick_channels(['F3-REF','F4-REF','C3-REF','C4-REF','O1-REF','O2-REF'])  # Select a subset of EEG channels
    
    
    data = eeg.get_data() 

    info = eeg.info
    channels = eeg.ch_names
    
    EEG_data = raw[8:14,:]
    chan = ['F3','F4','C3','C4','O1','O2']
    
