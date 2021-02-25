import numpy as np
import pandas as pd
import mne
import seaborn as sns
from matplotlib import pyplot as plt
import csv
import os
import datetime
import hrvanalysis as hrv
from scipy import signal

#let's just do one file at a time for this one 
FILE = '/Users/ajsimon/Dropbox (Personal)/Data/multimodal biosensing/raw data/001_AJ_10_26_2020.csv'

SUBID = 'AJS_10_26_2020'

#labels of electrodes in enobio montage -- DO NOT CHANGE THIS OR THE ORDER OF THESE ELECS
ELECS = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2','EOG']

# Load in the raw data -- this contains EEG, HRV, and EDA all in the same file. The EEG has a different srate from the EDA/HRV data, so we'll need to account for that difference later in this script
data = np.loadtxt(FILE,dtype = 'str',delimiter = ',',skiprows=1) 

headers = data[0,:]

### This first section will process the HRV and GSR data ###

#find the column that contains the HR data
HR_ix = [n for n, x in enumerate(headers) if 'Internal ADC A13 PPG RAW' in x]

HR_vec = data[1:,HR_ix]

#remove blank rows that exist because of the higher EEG srate
HR_vec = HR_vec[~np.all(HR_vec == '', axis=1)]
        
#need to convert to float for 'find_peaks' to work
HR_array = np.asarray(HR_vec)
HR_array = HR_array.astype(float)

#find the column that contains GSR conductance
GSR_ix = [n for n, x in enumerate(inheader) if 'GSR Conductance CAL' in x]
GSR_vec = data[1:,GSR_ix]

#remove blank rows that exist because of the higher EEG srate
GSR_vec = GSR_vec[~np.all(GSR_vec == '', axis=1)]
        
#need to convert to float for 'find_peaks' to work
GSR_array = np.asarray(GSR_vec)
GSR_array = GSR_array.astype(float)

#create time series 
IBI_raw,_ = signal.find_peaks(HR_array[:,0], distance=80)   #distance=65

i, j = 0, 1
sampling_rate = 128
IBI_ts = []
        
#convert time series of HR peaks to IBIs (in seconds units)
while i < len(IBI_raw) and j < len(IBI_raw): #prevent it from surpassing the end of the peaks array
    ibi = ((IBI_raw[j] - IBI_raw[i]) / sampling_rate) * 1000 #where i starts at the 0th peak and j starts at the first peak
    IBI_ts.append(ibi) #peak 1 - peak 0/ sampling rate = first element of ibi array
    i += 1
    j += 1
            
#remove first and last 5 IBIs
IBI_ts = IBI_ts[4:len(IBI_ts)-5]
        
### Clean up IBI time series -- here we use a threshold of 4 stdev to remove IBIs, but this can be adjusted if the user wants
meanIBI = np.mean(IBI_ts)
stdIBI = np.std(IBI_ts)
        
IBI_cleaned = []
outliers = 0
        
for element in IBI_ts:
    if element < (meanIBI + (4 * stdIBI)) and (element > meanIBI - (4 * stdIBI)):
        IBI_cleaned.append(element)
    else:
        outliers += 1
percent_removed = (outliers/len(IBI_ts)) * 100;
        
#if we're removing more than 20% of the data then something is wrong and this file should be flagged
if percent_removed < 20:
            
    time_domain_features = hrv.get_time_domain_features(IBI_cleaned)
    geometrical_features = hrv.get_geometrical_features(IBI_cleaned)
    frequency_domain_features = hrv.get_frequency_domain_features(IBI_cleaned)
    csi_cvi_features = hrv.get_csi_cvi_features(IBI_cleaned)
    poincare_plot_features = hrv.get_poincare_plot_features(IBI_cleaned)
    sampen = hrv.get_sampen(IBI_cleaned)
            
    td_keys = list(time_domain_features.keys())
    geom_keys = list(geometrical_features.keys())
    frequency_keys = list(frequency_domain_features.keys())
    csi_keys = list(csi_cvi_features.keys())
    poincare_keys = list(poincare_plot_features.keys())
    samp_keys = list(sampen.keys())
    
    #format header of output
    header = []
    ID = ['ID']
                
    GSR_meanh = ['GSR_mean']  
    GSR_maxh = ['GSR_max']
    GSR_minh = ['GSR_min']
    GSR_rangeh = ['GSR_range']
    GSR_slopeh = ['GSR_slope']
      
    header = td_keys + geom_keys + frequency_keys + csi_keys + poincare_keys + samp_keys + GSR_meanh + GSR_maxh + GSR_minh + GSR_rangeh + GSR_slopeh
                
    output = []
    output.append(header)
                 
    td_values = []
    for key in td_keys:
        value = time_domain_features[key]
        td_values.append(value)
            
    geom_values = []
    for key in geom_keys:
        value = geometrical_features[key]
        geom_values.append(value)

    frequency_values = []
    for key in frequency_keys:
        value = frequency_domain_features[key]
        frequency_values.append(value)

    csi_values = []
    for key in csi_keys:
        value = csi_cvi_features[key]
        csi_values.append(value)

    poincare_values = []
    for key in poincare_keys:
        value = poincare_plot_features[key]
        poincare_values.append(value)

    samp_values = []
    for key in samp_keys:
        value = sampen[key]
        samp_values.append(value)
            
    if not GSR_ix:
        subData = td_values + geom_values + frequency_values + csi_values + poincare_values + samp_values
        output.append(subData)
            
    # THIS IS WHERE I ADD GSR COMPUTATONS -- GOTTA FIGURE OUT EXACTLY WHAT I WANT TO COMPUTE AND HOW/WHY IT'S DONE
    else:


        
        
        
        
        
        
        
        
### Now we process EEG data ###