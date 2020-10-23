#### THIS SHOULD BE USED AFTER PREPROCESSING TO CORRECT FOR ANY HYPNOGRAM-PSG LENGTH MISMATCHES

#### DONT USE THIS AS A STAND ALONE SCRIPT. IT WONT WORK AND YOU'LL EITHER GET FRUSTRATED OR BREAK YOUR DATA

#### YOU SHOULD USE THIS CODE WHILE MUCH OF THE INFO FROM 'PSG_PREPROCESSING.PY' IS IN YOUR WORKSPACE -- OR AT LEAST THE VARIABLES CREATED BEFORE THE LOOP

#### THERE ARE DIFFERENT CHUNKS OF CODE IN HERE THAT DO DIFFERENT THINGS DEPENDING ON WHAT THE PARTICULAR PROBLEM IS LEADING TO THE DATA DIMENSION MISMATCH. I HAVE ADDED COMMENTS ABOVE EACH MAJOR CHUNK OF CODE THAT CAN HOPFULLY HELP YOU FIGURE OUT WHAT IS DOING WHAT. 


# What is the index of the participant in FILES (filelist from 'psg_preprocessing.py')
f = 42
        
eeg = mne.io.read_raw_fif(FILES[f], preload=True)
hypnogram = np.genfromtxt(HYPNO[f], delimiter=',',dtype=str,'formats': ('S2'))

# Do this if you are not seeing '-1' in your hypnogram -- this was a weird formatting issue that some participants had. 
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


#if this breaks, then the issue was a PSG-hypnogram length mismatch -- skip to next section
fig = yasa.plot_spectrogram(F3, SF, hypno=hypnogram, fmax=30, cmap='Spectral_r', trimperc=5)

Hyp_fname = HYPNO[f][np.s_[pathsep+1:len(FILES[f])]]

hypno_outfile = open(OUTPATH + Hyp_fname,'w')
with hypno_outfile:
    writer = csv.writer(hypno_outfile,delimiter=' ')
    writer.writerows(hypno_with_art)
    
    
############### length mismatch correction ##############
#how long is the PSG and the hypnogram? 
PSGlen = len(data[0])
Hyplen = len(newHyp)
    
# if data is larger than hypnogram, figure out which part of the data you need to delete, and indicate in the column indexes below
# IMPORTANT: visually inspect to make sure you deleted the right stuff before saving
datan = np.delete(data,np.s_[3812200:4117800],1)    #the range of these numbers should equal PSGlen - Hyplen

F3 = []
F3 = datan[0,:]
fig = yasa.plot_spectrogram(F3, SF, hypno=hypnogram, fmax=30, cmap='Spectral_r', trimperc=5)
    
    
    
# if they look aligned now, then save it! otherwise, try removing other sections of PSG data 
    
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