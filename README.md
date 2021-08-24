The code in this pipeline is designed to preprocess overnight EEG data and compute metrics related to sleep health. This pipeline can be easily modified to take in EEG data from multiple different file extentions (e.g., bdf, edf) and many different montages. However, at this moment it will only process and compute summary statistics on electrodes F3, F4, C3, C4, O1, and O2. This can be changed by the user. 

Files in the subfolder '\PreProcessing' are all devoted to preprocessing data. The data that they were designed to preprocess is specified in the filename, and are listed below:  
    'EO_EC_preprocessing' => processes resting state eyes open and eyes closed data
    'preprocess_AD_overnight' => processes overnight data from the Alzheimer's cohort
    'Preprocess_MSLT' => processes the MSLT data
    'PSG_preprocessing' => processes the data from the PSP, CBS, MCI, and FTD cohort
    'Preprocessing_visinspect' => this is designed to visually inspect each individual file.
    
All files should be visually inspected after preprocessing to ensure that it was done correctly!!

The preprocessing scripts will also output a file called 'artifact rejection info' and 'failures'. 
Artifact rejection info contains information about what percentage of data for each patient was rejected. This should be inspected -- if a patient had 0 artifacts or too many artifacts, then that indicates that the artifact rejection threshold should be changed for that patient (this can be done at the top of each script -- this is coded by the variable 'thresh'). 
'Failures' indicates which files the pipeline did not work for. This could be caused by any number of reasons (e.g., the file had different electrode names than what the script looks for, there was no data in the file (was just a header), etc...). The user should pay attention to this file after the script finishes running and figure out why any files (if there are any at all) failed. 

The files in the subfolder '/Computations' are all devoted to computing summary statistics to run stats on. Anything with the letters 'tf' in it computes to power spectrum of the data you feed it and stores that info for each electrode in the output folder. 'Spindles' will compute spindle stats, 'oof' stands for 'one over f', which will compute the AUC on the 1/f spectrum (a measure reflecting total electricity reaching the scalp). Those are the important ones, though there are a few others in there.

The stage that you want these scripts to compute statistics for can be specified in lines that look like this: where_during_sleep_wakes = np.isin(during_sleep_hypno, ['0']) 
The stage can be changed by changing the '0' (which is wake) to '1', '2', '3', or '4' ('4' here is for REM). 

The subfolder 'DataVisualization' has a few scripts in it, but really the only one that is important for the PSG sleep studies is the one called 'barGraphs_with_points'. This will make a bar graph with each individual data point overlayed on it. In order for this to work, you need to input the individual data values into the line 'y = np.array([([21.11,55.82,15.91,51.2,16.02,21.74,46.38]),np.array([35.67,103.51,25.34,99.91,21.52,37.66,89.34])])'. This example will yield 2 bars -- one with the first np array's data overlayed and the second with the second np array's data. This one can be a bit confusing for a novice with python because the number of elements in 'x' and 'colors' need to be the same as the number of arrays specified in 'y'.  

Feel free to email me at asimon445@gmail.com or alexander.simon@ucsf.edu if you have any questions. 
