import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import pickle
from math import sqrt

import os
import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------
# Read in the input data frames for training.
# Reading in files for the 9500 events.
# ------------------------------------------------------------

col_names = ['LT_3L', 'HT_3L', 'njets', 'j0pT', 'MET_3L', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'Mos1_3L', 'Mos2_3L', 'trilepM', 'mT0', 'mT1', 'mT2', 'highmT_3L', '3lmT_3L', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'pT0_3L', 'pT1_3L', 'pT2_3L' , 'pT_tau_3L']
cols = np.arange(0,39)

text_path = '../inputs'

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


#Concat all mass points, and pick out only the input variables used to train SOM
AllTrainEv = pd.concat([M500_9500Ev, M1000_9500Ev, M1500_9500Ev])
AllTrainEv = AllTrainEv[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]

#Separating the values and the label column
Train_val_UN = AllTrainEv[AllTrainEv.columns[:-1]].values #UN stands for un-normalized
Train_val = (Train_val_UN - np.mean(Train_val_UN, axis=0)) / np.std(Train_val_UN, axis=0) #Normalized
Train_lab = AllTrainEv['Label'].values



#-----------------------Plot variables----------------------
def PlotVars(mybins, plotnames, pp):
    for pname in plotnames:
        plt.figure(figsize=(20,20))
        plt.hist(M500_9500Ev[pname], bins = mybins, histtype = 'step', label = 'M500', linewidth = 3, color = 'teal', density = False, log = False)
        plt.hist(M1000_9500Ev[pname], bins = mybins, histtype = 'step', label = 'M1000', linewidth = 3, color = 'blue', density = False, log = False)
        plt.hist(M1500_9500Ev[pname], bins = mybins, histtype = 'step', label = 'M1500', linewidth = 3, color = 'red', density = False, log = False)
        plt.legend(loc = 'upper right', fontsize=30)
        plt.xlabel(pname, fontsize=30)
        plt.ylabel('Events', fontsize=30)
        #plt.title(pname, fontsize=12)
        plt.savefig(pp, format = 'pdf')
        plt.close()


#-----------------------------------------------------------------

#---------------------SepScore function-------------------------
def SepScore(df, som_dim0, som_dim1):
    tot500 = 0
    tot1000 = 0
    tot1500 = 0

    arr1 = []
    
    for i in np.arange(0, som_dim0):
        for j in np.arange(0, som_dim1):
            c500 = ((df['w_x'] == i) & (df['w_y'] == j) & (df['Label'] == 0)).sum() #Finds number of M500 points in one neuron
            c1000 = ((df['w_x'] == i) & (df['w_y'] == j) & (df['Label'] == 1)).sum() #Finds number of M1000 points in one neuron
            c1500 = ((df['w_x'] == i) & (df['w_y'] == j) & (df['Label'] == 2)).sum() #Finds number of M1500 points in one neuron
            tot500 += c500
            tot1000 += c1000
            tot1500 += c1500
            arr1.append([i, j, c500, c1000, c1500])

    arr1 = pd.DataFrame(arr1, columns = ['w_x', 'w_y', 'count500', 'count1000', 'count1500'])

    #Score for each neuron: Non-absolute
    arr1['score500'] = arr1.apply(lambda row:(row.count500 - row.count1000 - row.count1500)/(row.count500 + row.count1000 + row.count1500), axis=1)
    arr1['score1000'] = arr1.apply(lambda row:(row.count1000 - row.count500 - row.count1500)/(row.count500 + row.count1000 + row.count1500), axis=1)
    arr1['score1500'] = arr1.apply(lambda row:(row.count1500 - row.count500 - row.count1000)/(row.count500 + row.count1000 + row.count1500), axis=1)
    
    #nonEmptyNeurons
    NEN = ((arr1['count500'] + arr1['count1000'] + arr1['count1500']>0)).sum()

    #Combining Scores
    SepScore500 = arr1['score500'].sum()
    SepScore500 = SepScore500/NEN
    
    SepScore1000 = arr1['score1000'].sum()
    SepScore1000 = SepScore1000/NEN
    
    SepScore1500 = arr1['score1500'].sum()
    SepScore1500 = SepScore1500/NEN

    return SepScore500,SepScore1000,SepScore1500



#------------------------------------------------------------------------------------
