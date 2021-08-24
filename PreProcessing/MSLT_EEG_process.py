import glob
import numpy as np
import mne
import yasa
from matplotlib import pyplot as plt
import matplotlob as mat



# step 1: structure the files such that each PT gets a folder with .edfs and hypnograms from all trials inside
# step 2: count number of folders within a given root folder (this will indicate number of subjects)
# step 3: loop through all folders

# step 4: within each iteration of PTs, create lists of all .edfs and all hypnograms 
# step 5: nested loop, iterate through all trials (should be 5, but don't hard code this just in case it's not)
    # check it like:
    # t = 0
    # tnums = ['T1','T2','T3','T4','T5']
    # while tm == 0:
        # trial = something analogous to strmatch(tnums[t],FILES[f])
        
        # step 6: load file, remove extra info, bandpass filter, rereference (if possible), downsample (not necessarily in this order)
        # step 7: load hypnogram, calculate sleep latency, whether they fell asleep or not (binary), length of time asleep, and stages they got into
        # compute fft of cleaned EEG data (wake and sleep separately -- if no sleep then NaN in output)
        # ***Put each trial in a new column for each subject, so each pt will have "T1_wake_alpha", etc...

PATH = 