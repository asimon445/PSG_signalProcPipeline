PATH = '/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/preprocessed/'

#create list of hypnogram files  
HYPNO = glob.glob(PATH + '*.csv')
HYPNO.sort()

# this is for indexing the last '/' in the path so that we can pull the filename easily for each participant
sep = '/'
def find(PATH, sep):
    return [i for i, ltr in enumerate(PATH) if ltr == sep]

pathsep = list(find(PATH,sep))
pathsep = pathsep[len(pathsep)-1]

sf = 100.


#find continuous regions of WASO (start and stop indexes) in sleep data
def contiguous_regions(condition):
    idx = []
    i = 0
    while i < len(condition):
        x1 = i + condition[i:].argmax()
        try:
            x2 = x1 + condition[x1:].argmin()
        except:
            x2 = x1 + 1
        if x1 == x2:
            if condition[x1] == True:
                x2 = len(condition)
            else:
                break
        idx.append( [x1,x2] )
        i = x2
    return idx


## initialize for loop
for f, file in enumerate(HYPNO):

    if (f != 41):
        Hyp_fnum= HYPNO[f][np.s_[pathsep+1:pathsep+6]]   
            
        hypnogram = []
        hypnogram = np.loadtxt(fname = HYPNO[f],dtype = 'str',delimiter = ',')  
        hypnogram = hypnogram.astype('U3')
            
        for r in range(len(hypnogram)):
            if hypnogram[r] == '- 1':
                hypnogram[r] = '-1'
            elif hypnogram[r] == '2 1':
                hypnogram[r] = '2'
            elif hypnogram[r] == '-':
                hypnogram[r] = '-1'
                    
        sf = 100.
        times = np.arange(hypnogram.size) / sf
        
        # index locations within hypnogram where participant is in each stage
        where_art = np.isin(hypnogram, ['-1'])
        where_awake = np.isin(hypnogram, ['0']) 
        where_1 = np.isin(hypnogram, ['1'])
        where_2 = np.isin(hypnogram, ['2'])
        where_3 = np.isin(hypnogram, ['3'])
        where_REM = np.isin(hypnogram, ['4'])
        
        #index start and stop times for each time they enter a stage
        art_bouts = contiguous_regions(where_art)
        wake_bouts = contiguous_regions(where_awake)
        S1_bouts = contiguous_regions(where_1)
        S2_bouts = contiguous_regions(where_2)
        S3_bouts = contiguous_regions(where_3)
        REM_bouts = contiguous_regions(where_REM)
        
        #create a 3rd column in the variable containing start/stop times for each stretch of each stage containing info about how long they stayed in each stage
        art_len = np.empty((np.shape(art_bouts)[0], 1))
        for i in range(np.shape(art_bouts)[0]):
            art_len[i] = art_bouts[i][1] - art_bouts[i][0]
        
        if len(art_bouts) > 0:
            art_bouts = np.append(art_bouts, art_len, axis=1)
        
        wake_len = np.empty((np.shape(wake_bouts)[0], 1))
        for i in range(np.shape(wake_bouts)[0]):
            wake_len[i] = wake_bouts[i][1] - wake_bouts[i][0]
        
        if len(wake_bouts) > 0:
            wake_bouts = np.append(wake_bouts, wake_len, axis=1)
        
        S1_len = np.empty((np.shape(S1_bouts)[0], 1))
        for i in range(np.shape(S1_bouts)[0]):
            S1_len[i] = S1_bouts[i][1] - S1_bouts[i][0]
        
        if len(S1_bouts) > 0:
            S1_bouts = np.append(S1_bouts, S1_len, axis=1)
        
        S2_len = np.empty((np.shape(S2_bouts)[0], 1))
        for i in range(np.shape(S2_bouts)[0]):
            S2_len[i] = S2_bouts[i][1] - S2_bouts[i][0]
        
        if len(S2_bouts) > 0:
            S2_bouts = np.append(S2_bouts, S2_len, axis=1)
        
        S3_len = np.empty((np.shape(S3_bouts)[0], 1))
        for i in range(np.shape(S3_bouts)[0]):
            S3_len[i] = S3_bouts[i][1] - S3_bouts[i][0]
        
        if len(S3_bouts) > 0:
            S3_bouts = np.append(S3_bouts, S3_len, axis=1)
        
        REM_len = np.empty((np.shape(REM_bouts)[0], 1))
        for i in range(np.shape(REM_bouts)[0]):
            REM_len[i] = REM_bouts[i][1] - REM_bouts[i][0]
        
        if len(REM_bouts) > 0:
            REM_bouts = np.append(REM_bouts, REM_len, axis=1)
        
        
        # add a 4th column to the variable containing start/stop times for each stretch of each stage so that we can distinguish them after concatenating them all together
        empty_wake_ix = np.empty((np.shape(wake_bouts)[0], 1))
        for i in range(np.shape(empty_wake_ix)[0]):
            empty_wake_ix[i] = '0'
        wake_bouts = np.append(wake_bouts, empty_wake_ix, axis=1)        
        
        if len(art_bouts) > 0:
            empty_art_ix = np.empty((np.shape(art_bouts)[0], 1))
            for i in range(np.shape(empty_art_ix)[0]):
                empty_art_ix[i] = '-1'      
            art_bouts = np.append(art_bouts, empty_art_ix, axis=1)
        
        if len(S1_bouts) > 0:
            empty_S1_ix = np.empty((np.shape(S1_bouts)[0], 1))
            for i in range(np.shape(empty_S1_ix)[0]):
                empty_S1_ix[i] = '1'
            S1_bouts = np.append(S1_bouts, empty_S1_ix, axis=1)
        
        if len(S2_bouts) > 0:
            empty_S2_ix = np.empty((np.shape(S2_bouts)[0], 1))
            for i in range(np.shape(empty_S2_ix)[0]):
                empty_S2_ix[i] = '2'
            S2_bouts = np.append(S2_bouts, empty_S2_ix, axis=1)
        
        if len(S3_bouts) > 0:
            empty_S3_ix = np.empty((np.shape(S3_bouts)[0], 1))
            for i in range(np.shape(empty_S3_ix)[0]):
                empty_S3_ix[i] = '3'
            S3_bouts = np.append(S3_bouts, empty_S3_ix, axis=1)
        
        if len(REM_bouts) > 0:
            empty_REM_ix = np.empty((np.shape(REM_bouts)[0], 1))
            for i in range(np.shape(empty_REM_ix)[0]):
                empty_REM_ix[i] = '4'
            REM_bouts = np.append(REM_bouts, empty_REM_ix, axis=1)
        
        #concatenate them all together
        if len(art_bouts) > 0:
            all_stages = np.append(wake_bouts, art_bouts, axis=0)
        else:
            all_stages = wake_bouts
        
        if len(S1_bouts) > 0:
            all_stages = np.append(all_stages, S1_bouts, axis=0)
            
        if len(S2_bouts) > 0:
            all_stages = np.append(all_stages, S2_bouts, axis=0)
        
        if len(S3_bouts) > 0:
            all_stages = np.append(all_stages, S3_bouts, axis=0)
        
        if len(REM_bouts) > 0:
            all_stages = np.append(all_stages, REM_bouts, axis=0)
        
        sorted_stages = all_stages[all_stages[:,0].argsort()]
        
        relevant_only = sorted_stages[:,2:4]
        
        #save it! one file per subject
        outfile = open('/Users/ajsimon/Dropbox (Personal)/Data/Overnight PSG/data/hypnogram summary/' + Hyp_fnum + ' hypnogram summary.csv','w')
        with outfile:
            writer = csv.writer(outfile,delimiter=',')
            writer.writerows(relevant_only)
        
        
        