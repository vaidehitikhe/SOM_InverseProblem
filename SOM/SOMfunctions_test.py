import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, MaxNLocator
import pickle
from math import sqrt
from sklearn.metrics import roc_curve, roc_auc_score
import csv

import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Read in the input data frames for testing.
# Reading in files for the 2000 events.
# ------------------------------------------------------------

col_names = ['LT_3L', 'HT_3L', 'njets', 'j0pT', 'MET_3L', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'Mos1_3L', 'Mos2_3L', 'trilepM', 'mT0', 'mT1', 'mT2', 'highmT_3L', '3lmT_3L', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'pT0_3L', 'pT1_3L', 'pT2_3L' , 'pT_tau_3L']
cols = np.arange(0,39)

text_path = '../inputs'

#M500
M500_2kEv = pd.read_csv(text_path + '/M500/M500_2000Ev_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M500_2kEv['pT0j0pT'] = M500_2kEv['pT0_3L'] + M500_2kEv['j0pT']
M500_2kEv['Label'] = 0

#M1000
M1000_2kEv = pd.read_csv(text_path + '/M1000/M1000_2000Ev_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1000_2kEv['pT0j0pT'] = M1000_2kEv['pT0_3L'] + M1000_2kEv['j0pT']
M1000_2kEv['Label'] = 1

#M1500
M1500_2kEv = pd.read_csv(text_path + '/M1500/M1500_2000Ev_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1500_2kEv['pT0j0pT'] = M1500_2kEv['pT0_3L'] + M1500_2kEv['j0pT']
M1500_2kEv['Label'] = 2

#SM
#TTZ
TTZ_1kEv = pd.read_csv(text_path + '/SM/TTZ_1k_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
TTZ_1kEv['pT0j0pT'] = TTZ_1kEv['pT0_3L'] + TTZ_1kEv['j0pT']
#WZ
WZ_1kEv = pd.read_csv(text_path + '/SM/WZ_1k_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
WZ_1kEv['pT0j0pT'] = WZ_1kEv['pT0_3L'] + WZ_1kEv['j0pT']
#SM
SM_2kEv = pd.concat([TTZ_1kEv, WZ_1kEv])
SM_2kEv['Label'] = 4

#----------------------------------------------------------------------------------------




#------------------------4 Cases using 500Ev----------------------------------

#M500
M500_500Ev = pd.read_csv(text_path + '/M500/M500_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M500_500Ev['pT0j0pT'] = M500_500Ev['pT0_3L'] + M500_500Ev['j0pT']
M500_500Ev['Label'] = 0
#M500_500Ev = M500_2kEv.sample(frac=0.25);

#M1000
M1000_500Ev = pd.read_csv(text_path + '/M1000/M1000_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1000_500Ev['pT0j0pT'] = M1000_500Ev['pT0_3L'] + M1000_500Ev['j0pT']
M1000_500Ev['Label'] = 1
#M1000_500Ev = M1000_2kEv.sample(frac=0.25);

#M1500
M1500_500Ev = pd.read_csv(text_path + '/M1500/M1500_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1500_500Ev['pT0j0pT'] = M1500_500Ev['pT0_3L'] + M1500_500Ev['j0pT']
M1500_500Ev['Label'] = 2
#M1500_500Ev = M1500_2kEv.sample(frac=0.25);

#M750
M750_500Ev = pd.read_csv(text_path + '/M750/M750_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M750_500Ev['pT0j0pT'] = M750_500Ev['pT0_3L'] + M750_500Ev['j0pT']
M750_500Ev['Label'] = 3

#TTZ
TTZ_500Ev = pd.read_csv(text_path + '/SM/TTZ_sample500.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
TTZ_500Ev['pT0j0pT'] = TTZ_500Ev['pT0_3L'] + TTZ_500Ev['j0pT']
#TTZ_500Ev = TTZ_1kEv.sample(frac=0.5)
TTZ_500Ev['Label'] = 4
#TTZ_500Ev = TTZ_500Ev.sample(frac=0.5)


#WZ
WZ_500Ev = pd.read_csv(text_path + '/SM/WZ_sample500.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
WZ_500Ev['pT0j0pT'] = WZ_500Ev['pT0_3L'] + WZ_500Ev['j0pT']
#WZ_500Ev = WZ_1kEv.sample(frac=0.5)
WZ_500Ev['Label'] = 4
#WZ_500Ev = WZ_500Ev.sample(frac=0.5)


#SM
SM_cases = pd.concat([TTZ_500Ev, WZ_500Ev])

#M2500
M2500_500Ev = pd.read_csv(text_path + '/M2500/M2500_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M2500_500Ev['pT0j0pT'] = M2500_500Ev['pT0_3L'] + M2500_500Ev['j0pT']
M2500_500Ev['Label'] = 5

#----------------------------------------------------------------------------------------
#M500
M500_9500Ev = pd.read_csv(text_path + '/M500/M500_9500Ev_train.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M500_9500Ev['pT0j0pT'] = M500_9500Ev['pT0_3L'] + M500_9500Ev['j0pT']
M500_9500Ev['Label'] = 0

#M1000
M1000_9500Ev = pd.read_csv(text_path + '/M1000/M1000_9500Ev_train.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1000_9500Ev['pT0j0pT'] = M1000_9500Ev['pT0_3L'] + M1000_9500Ev['j0pT']
M1000_9500Ev['Label'] = 1

#M1500
M1500_9500Ev = pd.read_csv(text_path + '/M1500/M1500_9500Ev_train.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1500_9500Ev['pT0j0pT'] = M1500_9500Ev['pT0_3L'] + M1500_9500Ev['j0pT']
M1500_9500Ev['Label'] = 2

#Get mean and sd
AllTrainEv = pd.concat([M500_9500Ev, M1000_9500Ev, M1500_9500Ev])
AllTrainEv = AllTrainEv[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]

#Separating the values and the label column
Train_val_UN = AllTrainEv[AllTrainEv.columns[:-1]].values
Train_mean = np.mean(Train_val_UN, axis = 0)
Train_sd = np.std(Train_val_UN, axis = 0)
Train_val_UN = pd.DataFrame(Train_val_UN)
Train_val_UN.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L']
#----------------------------------------------------------------------------------------




#-------------------------SepScore Function------------------------------------
def SepScore(df, som_dim0, som_dim1): 
    array1 = []
    tot500 = 0
    tot1000 = 0
    tot1500 = 0
    totSM = 0

    for i in np.arange(0, som_dim0):
        for j in np.arange(0, som_dim1):
            count500 = ((df['w_x'] == i) & (df['w_y'] == j) & (df['Label'] == 0)).sum() #Finds number of M500 points in one neuron
            count1000 = ((df['w_x'] == i) & (df['w_y'] == j) & (df['Label'] == 1)).sum() #Finds number of M1000 points in one neuron
            count1500 = ((df['w_x'] == i) & (df['w_y'] == j) & (df['Label'] == 2)).sum() #Finds number of M1500 points in one neuron
            countSM = ((df['w_x'] == i) & (df['w_y'] == j) & (df['Label'] == 4)).sum() #Finds the number of backgrounds(WZ+TTZ) in one neuron
            tot500 += count500 #Finds the total number of M500 points in the grid
            tot1500 += count1000 #Finds the total number of M1000 points in the grid
            tot1500 += count1500 #Finds the total number of M1500 points in the grid
            totSM += countSM #Finds the total number of background points in the grid
    
            array1.append([i, j, count500, count1000, count1500, countSM])
            
    array1 = pd.DataFrame(array1, columns = ['w_x', 'w_y', 'count500', 'count1000', 'count1500', 'countSM'])
            
    #Score for each neuron
    array1['Score500'] = array1.apply(lambda row:((row.count500 - row.count1000 - row.count1500 - row.countSM))/(row.count500 + row.count1000 + row.count1500 + row.countSM), axis=1)
    array1['Score1000'] = array1.apply(lambda row:((row.count1000 - row.count500 - row.count1500 - row.countSM))/(row.count500 + row.count1000 + row.count1500 + row.countSM), axis=1)
    array1['Score1500'] = array1.apply(lambda row:((row.count1500 - row.count500 - row.count1000 - row.countSM))/(row.count500 + row.count1000 + row.count1500 + row.countSM), axis=1)
    array1['ScoreSM'] = array1.apply(lambda row: ((row.countSM - row.count500 - row.count1000 - row.count1500)) / (row.count500 + row.count1000 + row.count1500 + row.countSM), axis=1)

    #NEN (NonEmptyNeurons)
    nonEmptyNeurons = ((array1['count500'] + array1['count1000'] + array1['count1500'] + array1['countSM']>0)).sum()

    #Combining SepScore
    #Sum the scores of each neuron and divide by the number of nonEmptyNeurons
    SepScore500 = array1['Score500'].sum()
    SepScore500 = SepScore500/nonEmptyNeurons
    
    SepScore1000 = array1['Score1000'].sum()
    SepScore1000 = SepScore1000/nonEmptyNeurons
    
    SepScore1500 = array1['Score1500'].sum()
    SepScore1500 = SepScore1500/nonEmptyNeurons

    SepScoreSM = array1['ScoreSM'].sum()
    SepScoreSM = SepScoreSM/nonEmptyNeurons 

    return SepScore500,SepScore1000,SepScore1500,SepScoreSM
    
#------------------------------------------------------------------------------------------------------------------

#-------------------------ClusterScore------------------------------------
def ClusterScore(dataframe, lox, hix, loy, hiy):
    df = pd.DataFrame(dataframe)

    clus = df.loc[ (df['w_x']>=int(lox)) & (df['w_x']<=int(hix)) & (df['w_y']>=int(loy)) & (df['w_y']<=int(hiy)) ]

    arr=[]

    for i in np.arange(int(lox),int(hix+1)):
        for j in np.arange(int(loy),int(loy+1)):
            num500 = ( (clus['w_x']==i) & (clus['w_y']==j) & (clus['Label']==0) ).sum()
            num1000= ( (clus['w_x']==i) & (clus['w_y']==j) & (clus['Label']==1) ).sum()
            num1500= ( (clus['w_x']==i) & (clus['w_y']==j) & (clus['Label']==2) ).sum()
            numSM  = ( (clus['w_x']==i) & (clus['w_y']==j) & (clus['Label']==4) ).sum()
            arr.append([i,j,num500,num1000,num1500,numSM])
            # for loops end here

    arr = pd.DataFrame(arr,columns=['w_x','w_y','num500','num1000','num1500','numSM'])

    arr['Score500'] =arr.apply(lambda row: (row['num500']-row['num1000']-row['num1500']-row['numSM'])/(row['num500']+row['num1000']+row['num1500']+row['numSM']) if (row['num500']+row['num1000']+row['num1500']+row['numSM'])>0 else -1, axis=1)
    arr['Score1000']=arr.apply(lambda row: (row['num1000']-row['num500']-row['num1500']-row['numSM'])/(row['num500']+row['num1000']+row['num1500']+row['numSM']) if (row['num500']+row['num1000']+row['num1500']+row['numSM'])>0 else -1, axis=1)
    arr['Score1500']=arr.apply(lambda row: (row['num1500']-row['num500']-row['num1000']-row['numSM'])/(row['num500']+row['num1000']+row['num1500']+row['numSM']) if (row['num500']+row['num1000']+row['num1500']+row['numSM'])>0 else -1, axis=1)
    arr['ScoreSM']=arr.apply(lambda row: (row['numSM']-row['num500']-row['num1000']-row['num1500'])/(row['num500']+row['num1000']+row['num1500']+row['numSM']) if (row['num500']+row['num1000']+row['num1500']+row['numSM'])>0 else -1, axis=1)
    arr['isFilled'] = arr.apply(lambda row: 1 if (row['num500']+row['num1000']+row['num1500']+row['numSM']) >0  else 0, axis=1)

    
    nonEmptyNeurons = ((arr['num500']+arr['num1000']+arr['num1500']+arr['numSM'] >0)).sum()

    sepScore500 = -1
    sepScore1000 = -1
    sepScore1500 = -1
    sepScoreSM = -1
    if( nonEmptyNeurons > 0):
        filled = arr[arr['isFilled'] == 1]
        sepScore500  = filled['Score500'].sum()
        sepScore500  = sepScore500 / nonEmptyNeurons
        sepScore1000 = filled['Score1000'].sum()
        #print(f'sepScore1000 = {sepScore1000}')
        sepScore1000 = sepScore1000 / nonEmptyNeurons
        sepScore1500 = filled['Score1500'].sum()
        sepScore1500 = sepScore1500 / nonEmptyNeurons
        sepScoreSM   = filled['ScoreSM'].sum()
        sepScoreSM  = sepScoreSM / nonEmptyNeurons
    
    return [round(sepScore500,3), round(sepScore1000,3), round(sepScore1500,3), round(sepScoreSM,3)]   
#------------------------------------------------------------------------------------------------------------------

#-------------------------caseFunct----------------------------------------
def caseFunct(case_num,w_x, w_y, case_arr, case_lab_arr, pdf, som_arr, rangeA, test_df):    

    w_x = np.array(w_x)
    w_y = np.array(w_y)
    w_xy = np.column_stack((w_x, w_y))

    case_df = np.column_stack((case_arr, w_x, w_y, case_lab_arr))
    case_df = pd.DataFrame(case_df)
    case_df.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'w_x', 'w_y', 'Label']
    
    color_list = {0: 'teal', 1: 'blue', 2: 'red', 3: 'purple', 4: 'black', 5: 'orange'}
    label_list = {0: 'M500', 1: 'M1000', 2: 'M1500', 3: 'M750', 4: 'SM', 5: 'M2500'}
    #marker_list = {0: 'o', 1: 'o', 2: 'o', 3: 'o', 4: 'x', 5: 'o'}
    marker_list = {0: 'o', 1: '^', 2: 's', 3: 'P', 4: 'x', 5: '*'}
    # -----------
    # first we plot the case points
    # -----------
    plt.figure(figsize=(20,20))
    plt.pcolor(som_arr, cmap = 'bone_r', alpha = 0.0)

    for j in np.unique(case_lab_arr):
        idx_label = case_lab_arr == j
        plt.scatter(w_x[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, w_y[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, s=300, c = 'xkcd:faded purple', label = label_list[j], marker = 'P')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1, alpha = 0.9)
    plt.tick_params(axis='both', labelsize=25)
    plt.xlabel('Neuron X', fontsize=35)
    plt.ylabel('Neuron Y', fontsize=35)
    plt.title(f'Case {case_num}',fontsize=35)
    plt.savefig(pdf, format = 'pdf')

    # -----------
    # Now we find the neighborhoods
    # -----------    
    ranAclus_arr = [[0,0,0,0] for _ in range(len(case_df))]
    for j in range(0,len(case_df)):
        ranAclus_arr[j][0] = case_df.iloc[j,8] - rangeA #w_x_lo
        ranAclus_arr[j][1] = case_df.iloc[j,8] + rangeA #w_x_hi
        ranAclus_arr[j][2] = case_df.iloc[j,9] - rangeA #w_y_lo
        ranAclus_arr[j][3] = case_df.iloc[j,9] + rangeA #w_y_hi

    for k in range(0, len(case_df)):
        for l in range(0,4):
            if(ranAclus_arr[k][l] < 0):
                ranAclus_arr[k][l] = 0
            if(ranAclus_arr[k][l] > 100):
                ranAclus_arr[k][l] = 100

    ranAclus_arr = np.array(ranAclus_arr)
    scores_ranA_arr = [[0,0,0,0] for _ in range(len(case_df))]
    for m in range(len(case_df)):
        scores_ranA_arr[m] = ClusterScore(test_df, ranAclus_arr[m][0], ranAclus_arr[m][1], ranAclus_arr[m][2], ranAclus_arr[m][3])
    
    
    scores_ranA_arr = np.array(scores_ranA_arr)
    medSS500A = np.median([row[0] for row in scores_ranA_arr])
    medSS1000A = np.median([row[1] for row in scores_ranA_arr])
    medSS1500A = np.median([row[2] for row in scores_ranA_arr])
    medSSSMA = np.median([row[3] for row in scores_ranA_arr])

    plt.figure(figsize=(20,20))
    plt.hist(scores_ranA_arr[: , 0], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{500}}} $" + f'({medSS500A:.3f})', linewidth = 3, color = 'xkcd:pinkish purple', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_arr[: , 1], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1000}}} $" + f'({medSS1000A:.3f})', linewidth = 3, color = 'xkcd:tangerine', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_arr[: , 2], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1500}}} $" + f'({medSS1500A:.3f})', linewidth = 3, color = 'xkcd:murky green', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_arr[: , 3], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{SM}}} $" + f'({medSSSMA:.3f})', linewidth = 3, color = 'black', density = False, log = False, alpha = 0.3)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc= 'upper right', fontsize=30)
    plt.xlabel(f'Regional SepScore ({2*rangeA+1} x {2*rangeA+1})', fontsize=30)
    plt.ylabel('Number of Clusters', fontsize=30)
    plt.tick_params(axis='both', which = 'major', labelsize=15)
    plt.title(f'Case {case_num}',fontsize=35)
    plt.savefig(pdf, format = 'pdf')

    #Case_lab, SS500, SS1000, SS1500, SSSM
    case_lab_arr2 = np.array(case_lab_arr)[:, np.newaxis]
    scores_labels_ranA = np.hstack([case_lab_arr2, scores_ranA_arr])
    scores_labels_ranA = pd.DataFrame(scores_labels_ranA)
    scores_labels_ranA.columns = ['Label', 'SS500', 'SS1000', 'SS1500', 'SSSM']
    
    #Cutting on the sepScoreSM < 0.6 for rangeA (+-2 around the point)
    scores_ranA_SMcut = [row for row in scores_ranA_arr if row[-1] < 0.6]
    scores_ranA_SMcut = np.array(scores_ranA_SMcut)
    medSS500Acut = np.median([row[0] for row in scores_ranA_SMcut])
    medSS1000Acut = np.median([row[1] for row in scores_ranA_SMcut])
    medSS1500Acut = np.median([row[2] for row in scores_ranA_SMcut])
    medSSSMAcut = np.median([row[3] for row in scores_ranA_SMcut])

    #Picking only those rows for which SSBkg <0.6 from scores_labels_ranA into scores_labels_ranA_Bkgcut
    scores_labels_ranA_SMcut = scores_labels_ranA[scores_labels_ranA['SSSM'] < 0.6]
    
    plt.figure(figsize=(20,20))
    plt.hist(scores_ranA_SMcut[: , 0], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{500}}} $" + f'({medSS500Acut:.3f})', linewidth = 3, color = 'xkcd:pinkish purple', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_SMcut[: , 1], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1000}}} $" + f'({medSS1000Acut:.3f})', linewidth = 3, color = 'xkcd:tangerine', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_SMcut[: , 2], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1500}}} $" + f'({medSS1500Acut:.3f})', linewidth = 3, color = 'xkcd:murky green', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_SMcut[: , 3], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label =r"${\mathrm{SepScore_{SM}}} $" + f'({medSSSMAcut:.3f})', linewidth = 3, color = 'black', density = False, log = False, alpha = 0.3)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(f'Regional SepScore ({2*rangeA+1} x {2*rangeA+1})', fontsize=30)
    plt.ylabel('Number of Clusters', fontsize=30)
    plt.legend(loc= 'upper right', fontsize=30)
    plt.tick_params(axis='both', which = 'major', labelsize=15)    
    plt.title(f'Case {case_num}',fontsize=35)
    #plt.legend(loc= 'upper right')
    plt.savefig(pdf, format = 'pdf')
#------------------------------------------------------------------------------------------------------------------

    
#-------------------------caseFunct_mod------------------------------------
def caseFunct_mod(w_x, w_y, case_arr, case_lab_arr, rangeA, test_df):    
    w_x = np.array(w_x)
    w_y = np.array(w_y)
    w_xy = np.column_stack((w_x, w_y))

    case_df = np.column_stack((case_arr, w_x, w_y, case_lab_arr))
    case_df = pd.DataFrame(case_df)
    case_df.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'w_x', 'w_y', 'Label']

    ranAclus_arr = [[0,0,0,0] for _ in range(len(case_df))]
    for j in range(0,len(case_df)):
        ranAclus_arr[j][0] = case_df.iloc[j,8] - rangeA #w_x_lo
        ranAclus_arr[j][1] = case_df.iloc[j,8] + rangeA #w_x_hi
        ranAclus_arr[j][2] = case_df.iloc[j,9] - rangeA #w_y_lo
        ranAclus_arr[j][3] = case_df.iloc[j,9] + rangeA #w_y_hi
        
    for k in range(0, len(case_df)):
        for l in range(0,4):
            if(ranAclus_arr[k][l] < 0):
                ranAclus_arr[k][l] = 0
            if(ranAclus_arr[k][l] > 100):
                ranAclus_arr[k][l] = 100
    
    ranAclus_arr = np.array(ranAclus_arr)
    scores_ranA_arr = [[0,0,0,0] for _ in range(len(case_df))]
    for m in range(len(case_df)):
        scores_ranA_arr[m] = ClusterScore(test_df, ranAclus_arr[m][0], ranAclus_arr[m][1], ranAclus_arr[m][2], ranAclus_arr[m][3]) #This is the regional sepscore (non-absolute)

    print()
        
    #Case_lab, SS500, SS1000, SS1500, SSBkg
    case_lab_arr2 = np.array(case_lab_arr)[:, np.newaxis]
    scores_labels_ranA = np.hstack([case_lab_arr2, scores_ranA_arr])
    scores_labels_ranA = pd.DataFrame(scores_labels_ranA)
    scores_labels_ranA.columns = ['Label', 'SS500', 'SS1000', 'SS1500', 'SSSM']
    print(f"case_lab_arr2: {len(case_lab_arr2)}")
    print(f"case_lab_arr: {len(case_lab_arr)}")
    print(f"scores_ranA_arr: {len(scores_ranA_arr)}")
    
    return scores_labels_ranA
#------------------------------------------------------------------------------------------------------------------


#-------------------------plotROC------------------------------------
def plotROC(df,pdf):

    # The df should have columns as  ['Label', 'SS500', 'SS1000', 'SS1500', 'SSSM']

    #---------------------------------------------------------------------    
    M500_S500 = df[df['Label']==0.0]['SS500']
    Mothers_S500=df[df['Label'].isin([1.0,2.0,4.0])]['SS500']
    sig = M500_S500
    bkg = Mothers_S500
    y_true = np.concatenate([np.ones(len(sig)),np.zeros(len(bkg))])
    y_scores = np.concatenate([sig,bkg])
    fpr500, tpr500, _ = roc_curve(y_true,y_scores)
    with open("FprTpr500.txt", "w") as f:
        writer = csv.writer(f, delimiter = ' ')
        writer.writerows(zip(fpr500, tpr500))
    print('fpr500, tpr500 written in a txt file!')
       
    auc500 = roc_auc_score(y_true,y_scores)
    #---------------------------------------------------------------------
    M1000_S1000 = df[df['Label']==1.0]['SS1000']
    Mothers_S1000=df[df['Label'].isin([0.0,2.0,4.0])]['SS1000']
    sig = M1000_S1000
    bkg = Mothers_S1000
    y_true = np.concatenate([np.ones(len(sig)),np.zeros(len(bkg))])
    y_scores = np.concatenate([sig,bkg])
    fpr1000, tpr1000, _ = roc_curve(y_true,y_scores)
    with open("FprTpr1000.txt", "w") as f:
        writer = csv.writer(f, delimiter = ' ')
        writer.writerows(zip(fpr1000, tpr1000))
    print('fpr1000, tpr1000 written in a txt file!')
    
    auc1000 = roc_auc_score(y_true,y_scores)
    #---------------------------------------------------------------------
    M1500_S1500 = df[df['Label']==2.0]['SS1500']
    Mothers_S1500=df[df['Label'].isin([0.0,1.0,4.0])]['SS1500']
    sig = M1500_S1500
    bkg = Mothers_S1500
    y_true = np.concatenate([np.ones(len(sig)),np.zeros(len(bkg))])
    y_scores = np.concatenate([sig,bkg])
    fpr1500, tpr1500, _ = roc_curve(y_true,y_scores)
    with open("FprTpr1500.txt", "w") as f:
        writer = csv.writer(f, delimiter = ' ')
        writer.writerows(zip(fpr1500, tpr1500))
    print('fpr1500, tpr1500 written in a txt file!')
    
    auc1500 = roc_auc_score(y_true,y_scores)
    #---------------------------------------------------------------------

    plt.figure(figsize=(20,20))
    plt.plot(fpr500, tpr500, label=f'M500 vs rest (AUC={auc500:.3f})', color='xkcd:pinkish purple', lw=2)
    plt.plot(fpr1000, tpr1000, label=f'M1000 vs rest (AUC={auc1000:.3f})', color='xkcd:tangerine', lw=2)
    plt.plot(fpr1500, tpr1500, label=f'M1500 vs rest (AUC={auc1500:.3f})', color='xkcd:murky green', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Diagonal 'random guess' line
    plt.xlabel('Background Efficiency')
    plt.ylabel('Signal Efficiency')
    plt.title('ROC from SOM')
    plt.legend(loc='lower right')
    plt.savefig(pdf, format='pdf')

    plt.figure(figsize=(20,20))
    plt.hist(M500_S500,bins=np.arange(-1.0,1.4,0.1),alpha=0.9,linewidth=3,color='xkcd:pinkish purple',density=True,log=False,histtype='step',label='M500')
    plt.hist(Mothers_S500,bins=np.arange(-1.0,1.4,0.1),alpha=0.4,linewidth=3,color='black',density=True,log=False,histtype='step',label='Others')
    plt.legend(loc= 'upper right', fontsize=15)
    plt.xlabel('Separation Score 500', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    plt.savefig(pdf, format='pdf')

    plt.figure(figsize=(20,20))
    plt.hist(M1000_S1000,bins=np.arange(-1.0,1.4,0.1),alpha=0.9,linewidth=3,color='xkcd:tangerine',density=True,log=False,histtype='step',label='M1000')
    plt.hist(Mothers_S1000,bins=np.arange(-1.0,1.4,0.1),alpha=0.4,linewidth=3,color='black',density=True,log=False,histtype='step',label='Others')
    plt.legend(loc= 'upper right', fontsize=15)
    plt.xlabel('Separation Score 1000', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    plt.savefig(pdf, format='pdf')

    plt.figure(figsize=(20,20))
    plt.hist(M1500_S1500,bins=np.arange(-1.0,1.4,0.1),alpha=0.9,linewidth=3,color='xkcd:murky green',density=True,log=False,histtype='step',label='M1500')
    plt.hist(Mothers_S1000,bins=np.arange(-1.0,1.4,0.1),alpha=0.4,linewidth=3,color='black',density=True,log=False,histtype='step',label='Others')
    plt.legend(loc= 'upper right', fontsize=15)
    plt.xlabel('Separation Score 1500', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    plt.savefig(pdf, format='pdf')

    
#------------------------------------------------------------------------------------------------------------------
