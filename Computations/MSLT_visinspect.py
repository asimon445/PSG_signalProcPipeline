# this if for visually inspecting MSLT data
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


PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/MSLT/preprocessed/'

SUBDIRS = glob.glob(PATH + '/*/')   #list subdirectories
SUBDIRS.sort()

SF = 100    #Define frequency to downsample to -- Nyquist rate is 90 Hz for resolving spectral power in frequencies up to 45 Hz (gamma = 30-45 Hz -- per Walsh et al., 2017) 
            #VERY IMPORTANT THAT THE SAMPLING RATE MUST BE KEPT ABOVE THIS
            #Raw data is sampled at 400 Hz -- we don't need to downsample but can if we want to reduce size of data


for s in range(len(SUBDIRS)):

    # this is for indexing the last '/' in the path so that we can pull the filename easily for each participant
    sep = '/'
    currdir = SUBDIRS[s]
    def find(currdir, sep):
        return [i for i, ltr in enumerate(currdir) if ltr == sep]

    pathseps = list(find(currdir,sep))
    pathsep = pathseps[len(pathseps)-1]   #this here is the position of the final '/'
    
    # generate lists of .edf and .xls files
    edfs = glob.glob(currdir + '*.fif')
    edfs.sort()
    
    xlss = glob.glob(currdir + '*.csv')
    xlss.sort()

    for f in range(len(edfs)):
        
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
        if xl_match > 0:

            subpathseps = list(find(SUBDIRS[s],sep))
            subpathsep = subpathseps[len(pathseps)-1]   #this here is the position of the final '/'

            Hyp_fname = xlss[xl_match][np.s_[pathsep+1:len(xlss[0])]]
            PSG_fname = edfs[f][np.s_[pathsep+1:len(edfs[0])]]
            
            #load PSG file
            eeg = mne.io.read_raw_fif(edfs[f], preload=True)
            data = eeg.get_data() 

            #convert data from Volts to µV
            data = data*1000000
            hypnogram = np.loadtxt(fname = xlss[xl_match],dtype = 'str',delimiter = ',')  
            
            for i in range(len(hypnogram)):
            
                if hypnogram[i] == '0 . 0':
                    
                    hypnogram[i] = '0'
                    
                elif hypnogram[i] == '- 1':
                    
                    hypnogram[i] = '-1'
            
                elif hypnogram[i] == '2 . 0':
                    
                    hypnogram[i] = '2'
                
                elif hypnogram[i] == '3 . 0':
                    
                    hypnogram[i] = '3'
                    
                elif hypnogram[i] == '4 . 0':
                    
                    hypnogram[i] = '4'
                
            hypnogram = hypnogram.astype('f')
            
            if data.shape[1] < hypnogram.shape[0]:
            
                hypnogram = np.delete(hypnogram,np.s_[data.shape[1]:hypnogram.shape[0]],1)
                
            fig = yasa.plot_spectrogram(data[0,:], SF, hypno=hypnogram, fmax=30, cmap='Spectral_r', trimperc=5)
            
        else:
            print('Fuck you, no hypnogram')
            
        f=f+1
