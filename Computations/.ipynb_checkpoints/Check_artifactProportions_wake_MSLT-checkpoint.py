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

PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/MSLT/preprocessed_1sec_windows_1point5/'

SUBDIRS = glob.glob(PATH + '/*/')   #list subdirectories
SUBDIRS.sort()

SF = 100    #Define frequency to downsample to -- Nyquist rate is 90 Hz for resolving spectral power in frequencies up to 45 Hz (gamma = 30-45 Hz -- per Walsh et al., 2017) 
            #VERY IMPORTANT THAT THE SAMPLING RATE MUST BE KEPT ABOVE THIS
            #Raw data is sampled at 400 Hz -- we don't need to downsample but can if we want to reduce size of data
        
#format output header  -- descriptives
Output = np.zeros((len(SUBDIRS)+1,4), dtype=np.int)
Output = Output.astype('U30')
Output[0,0] = 'Subject'
Output[0,1] = 'Artifact time / Wake time'
Output[0,2] = 'Total wake time'
Output[0,3] = 'Total artifact time'

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

    wakelen = 0.
    artlen = 0.
    
    for f in range(len(fifs)):
        where_art = []
        where_wake = [] 
        
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
               
            
            Output[s+1,0] = PSG_fnum
            
            if isT1 > 0:
                    
                hypno = hypno.astype('str')
                                            
                #find samples in hypnogram that correspond to wake
                where_wake = np.isin(hypno, ['0.0'])  
                wakevec = hypno[where_wake]  

                #find samples in hypnogram that correspond to artifacts
                where_art = np.isin(hypno, ['-1.0'])  
                artvec = hypno[where_art]  
                
            elif isT2 > 0:
                
                hypno = hypno.astype('str')
                                            
                #find samples in hypnogram that correspond to wake
                where_wake = np.isin(hypno, ['0.0'])  
                wakevec = hypno[where_wake] 

                #find samples in hypnogram that correspond to artifacts
                where_art = np.isin(hypno, ['-1.0'])  
                artvec = hypno[where_art]  
                
                hyp_wake = hypno[where_wake]      
                hyp_art = hypno[where_art]   
                
            elif isT3 > 0:
                
                hypno = hypno.astype('str')
                                            
                #find samples in hypnogram that correspond to wake
                where_wake = np.isin(hypno, ['0.0'])  
                wakevec = hypno[where_wake] 
                artvec = hypno[where_art]  

                #find samples in hypnogram that correspond to artifacts
                where_art = np.isin(hypno, ['-1.0'])       
                
            elif isT4 > 0:
                
                hypno = hypno.astype('str')
                                            
                #find samples in hypnogram that correspond to wake
                where_wake = np.isin(hypno, ['0.0'])  
                wakevec = hypno[where_wake] 
                artvec = hypno[where_art]  

                #find samples in hypnogram that correspond to artifacts
                where_art = np.isin(hypno, ['-1.0'])  
    
            elif isT5 > 0:
                
                hypno = hypno.astype('str')
                                            
                #find samples in hypnogram that correspond to wake
                where_wake = np.isin(hypno, ['0.0'])  
                wakevec = hypno[where_wake] 

                #find samples in hypnogram that correspond to artifacts
                where_art = np.isin(hypno, ['-1.0'])  
                artvec = hypno[where_art]  
                
        wakelen = wakevec.shape[0] + wakelen
        artlen = artvec.shape[0] + artlen
        
    Output[s+1,0] = PSG_fnum
    Output[s+1,1] = artlen/wakelen
    Output[s+1,2] = wakelen/SF
    Output[s+1,3] = artlen/SF
    
outfile = open(PATH + 'Amount_of_awaketime.csv','w')
with outfile:
    writer = csv.writer(outfile,delimiter=',')
    writer.writerows(Output)