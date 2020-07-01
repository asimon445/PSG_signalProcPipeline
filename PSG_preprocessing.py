import yasa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(font_scale=1.2)
import mne
import glob

PATH = '/Users/ajsimon/Dropbox (Personal)/Overnight PSG/'

## IMPORTANT! The user must make sure that every patient with a PSG file also contains a corresponding hypnogram! 
## If there is anyone without one, staging will not work. Put those files in a separate folder

#create list of PSG files
FILES = glob.glob(PATH + '*.edf')

#create list of hypnogram files  
HYPNO = glob.glob(PATH + '*.csv')

SF = 100    #Define frequency to downsample to -- Nyquist rate is 90 Hz for resolving spectral power in frequencies up to 45 Hz (gamma = 30-45 Hz -- per Walsh et al., 2017) 
            #VERY IMPORTANT THAT THE SAMPLING RATE MUST BE KEPT ABOVE THIS
            #Raw data is sampled at 400 Hz

for f, file in enumerate(FILES):
    
    #load PSG file
    eeg = mne.io.read_raw_edf(FILES[f], preload=True)
    
    #load hypnogram for this same subject
    hypno = np.loadtxt(fname = FILES[f],dtype = 'str',delimiter = ',',skiprows=5)  
    
    picks = mne.pick_types(eeg.info, meg=False, eeg=True, eog=True)
    
    #Downsample to SF 
    eeg.resample(100)        
    
    #re-reference EEG to linked-mastoids, as opposed to the contralateral mastoid reference that the raw signal is referenced to
    eeg.set_eeg_reference(['M1-REF', 'M2-REF'])
    
    eeg.filter(0.3, 40)                    # Apply a bandpass filter from 0.3 to 40 Hz -- 
    
    
    #option 1: use an EOG defined-threshold for removing artifacts
    #if this is what we use, we'll keep the EOG. If not, then we'll remove it so that the covariance method is more likely to work
    
    #reject = dict(eog=150e-6)
    #epochs_params = dict(events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                     picks=picks, reject=reject, proj=True)
    
    #option 2: use the covariance-based method provided by YASA 
        # - this will in threory detect ocular and other artifacts, where EOG is more likely to just detect ocular. Option 2 is prob better
        
    eeg.pick_channels(['F3-REF','F4-REF','C3-REF','C4-REF','O1-REF','O2-REF'])  # Select a subset of EEG channels
    data = eeg.get_data() 
    
    art, zscores = yasa.art_detect(data, SF, window=5, method='covar', threshold=2)
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
    
