import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, MaxNLocator
import pickle
from math import sqrt
from minisom import MiniSom
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score
import csv

import os
import sys
import warnings
warnings.filterwarnings('ignore')


#Read in the text files

col_names = ['LT_3L', 'HT_3L', 'njets', 'j0pT', 'MET_3L', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'Mos1_3L', 'Mos2_3L', 'trilepM', 'mT0', 'mT1', 'mT2', 'highmT_3L', '3lmT_3L', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'pT0_3L', 'pT1_3L', 'pT2_3L' , 'pT_tau_3L']
cols = np.arange(0,39)




#-------------------------Train_9.5k-----------------------------------------------
text_path_train = '../text_files/text_files_final'

#M500
M500_9500Ev = pd.read_csv(text_path_train + '/M500/M500_9500Ev_train.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M500_9500Ev['pT0j0pT'] = M500_9500Ev['pT0_3L'] + M500_9500Ev['j0pT']
M500_9500Ev['Label'] = 0

#M1000
M1000_9500Ev = pd.read_csv(text_path_train + '/M1000/M1000_9500Ev_train.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1000_9500Ev['pT0j0pT'] = M1000_9500Ev['pT0_3L'] + M1000_9500Ev['j0pT']
M1000_9500Ev['Label'] = 1

#M1500
M1500_9500Ev = pd.read_csv(text_path_train + '/M1500/M1500_9500Ev_train.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
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




#-------------------------Test_2k----------------------------------------------------
text_path_test = '../text_files/text_files_final'

#M500
M500_2kEv = pd.read_csv(text_path_test + '/M500/M500_2000Ev_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M500_2kEv['pT0j0pT'] = M500_2kEv['pT0_3L'] + M500_2kEv['j0pT']
M500_2kEv['Label'] = 0

#M1000
M1000_2kEv = pd.read_csv(text_path_test + '/M1000/M1000_2000Ev_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1000_2kEv['pT0j0pT'] = M1000_2kEv['pT0_3L'] + M1000_2kEv['j0pT']
M1000_2kEv['Label'] = 1

#M1500
M1500_2kEv = pd.read_csv(text_path_test + '/M1500/M1500_2000Ev_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1500_2kEv['pT0j0pT'] = M1500_2kEv['pT0_3L'] + M1500_2kEv['j0pT']
M1500_2kEv['Label'] = 2

#SM
#TTZ
TTZ_1kEv = pd.read_csv(text_path_test + '/SM/TTZ_1k_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
TTZ_1kEv['pT0j0pT'] = TTZ_1kEv['pT0_3L'] + TTZ_1kEv['j0pT']
#WZ
WZ_1kEv = pd.read_csv(text_path_test + '/SM/WZ_1k_test.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
WZ_1kEv['pT0j0pT'] = WZ_1kEv['pT0_3L'] + WZ_1kEv['j0pT']
#SM
SM_2kEv = pd.concat([TTZ_1kEv, WZ_1kEv])
SM_2kEv['Label'] = 4

#----------------------------------------------------------------------------------------




#------------------------4 Cases using 500Ev----------------------------------
text_path = '../text_files/text_files_final'

#M500
M500_500Ev = pd.read_csv(text_path + '/M500/M500_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M500_500Ev['pT0j0pT'] = M500_500Ev['pT0_3L'] + M500_500Ev['j0pT']
M500_500Ev['Label'] = 0

#M1000
M1000_500Ev = pd.read_csv(text_path + '/M1000/M1000_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1000_500Ev['pT0j0pT'] = M1000_500Ev['pT0_3L'] + M1000_500Ev['j0pT']
M1000_500Ev['Label'] = 1

#M1500
M1500_500Ev = pd.read_csv(text_path + '/M1500/M1500_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M1500_500Ev['pT0j0pT'] = M1500_500Ev['pT0_3L'] + M1500_500Ev['j0pT']
M1500_500Ev['Label'] = 2

#M750
M750_500Ev = pd.read_csv(text_path + '/M750/M750_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M750_500Ev['pT0j0pT'] = M750_500Ev['pT0_3L'] + M750_500Ev['j0pT']
M750_500Ev['Label'] = 3

#TTZ
TTZ_500Ev = pd.read_csv(text_path + '/SM/TTZ_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
TTZ_500Ev['pT0j0pT'] = TTZ_500Ev['pT0_3L'] + TTZ_500Ev['j0pT']
TTZ_500Ev['Label'] = 4

#WZ
WZ_500Ev = pd.read_csv(text_path + '/SM/WZ_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
WZ_500Ev['pT0j0pT'] = WZ_500Ev['pT0_3L'] + WZ_500Ev['j0pT']
WZ_500Ev['Label'] = 4

#SM
SM_cases = pd.concat([TTZ_500Ev, WZ_500Ev])

#M2500
M2500_500Ev = pd.read_csv(text_path + '/M2500/M2500_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M2500_500Ev['pT0j0pT'] = M2500_500Ev['pT0_3L'] + M2500_500Ev['j0pT']
M2500_500Ev['Label'] = 5

#M600
M600_500Ev = pd.read_csv(text_path + '/M600/M600_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M600_500Ev['pT0j0pT'] = M600_500Ev['pT0_3L'] + M600_500Ev['j0pT']
M600_500Ev['Label'] = 6 #M600

#M800
M800_500Ev = pd.read_csv(text_path + '/M800/M800_500Ev_cases.txt', sep = ' ', index_col = None, usecols = cols, names = col_names)
M800_500Ev['pT0j0pT'] = M800_500Ev['pT0_3L'] + M800_500Ev['j0pT']
M800_500Ev['Label'] = 7

#----------------------------------------------------------------------------------------

#--------------Plotting----------------
#This function is specifically used for potting hist of input vars for those events which pass the SSSM<0.6 cut
def PlotVars(df, mybins, mybins1, plotnames, xlab, pdf, rang, case_num):
    for pname, lab in zip(plotnames, xlab):   
        plt.figure(figsize=(8,8))
        plt.hist(df[pname], bins = mybins, histtype = 'step', label = f'Case {case_num} Observed', linewidth = 3, color = 'xkcd:faded purple', density = True, log = False)
        plt.hist(M500_9500Ev[pname], bins = mybins1, histtype = 'step', label = r"${{m_{L} = 500\ GeV}}$", linewidth = 3, color = 'teal', density = True, log = False, alpha = 0.5)
        plt.hist(M1000_9500Ev[pname], bins = mybins1, histtype = 'step', label = r"${{m_{L} = 1000\ GeV}}$", linewidth = 3, color = 'blue', density = True, log = False, alpha = 0.5)
        plt.hist(M1500_9500Ev[pname], bins = mybins1, histtype = 'step', label = r"${{m_{L} = 1500\ GeV}}$", linewidth = 3, color = 'red', density = True, log = False, alpha = 0.5)
        ax = plt.gca()
        ax.set_xlim(left=0)   # Force x starts at 0
        ax.set_ylim(bottom=0)
        plt.legend(loc = 'upper right', fontsize=15)
        plt.xlabel(lab + f"  GeV", fontsize=20)
        plt.ylabel('Events', fontsize=20)
        #plt.title(pname, fontsize=12)
        plt.tick_params(axis='both', which = 'major', labelsize=15)
        plt.savefig(pdf, bbox_inches = 'tight', pad_inches=0, format = 'pdf')
        plt.close()
#---------------------------------


#-------------------------SepScore Function------------------------------------
def SepScore(array1, df, use, som_dim0, som_dim1): 
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

    plt.figure(figsize=(8.27, 11.69))
    txt_1 = f"-------------------------------Model Summary {use}---------------------------------" '\n' " " '\n' f"Dimension of SOM = {som_dim0} x {som_dim1}" '\n' " " '\n' f"Total neurons = {som_dim0 * som_dim1}, NonEmptyNeurons = {nonEmptyNeurons}"  '\n' " "  '\n' f"SepScore500 = {SepScore500:.3f}" '\n' f"SepScore1000 = {SepScore1000:.3f}" '\n' f"SepScore1500 = {SepScore1500:.3f}" '\n' f"SepScoreSM = {SepScoreSM:.3f}" '\n'
    plt.text(0.5, 0.90, txt_1, ha='center', va='top', wrap=True, fontsize=12)
    plt.axis('off')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



'''
#--------------------- Cluster Evaluation Function--------------------------------------------------------
def ClusterScore(dataframe, ClusterNum, lo_w_x, hi_w_x, lo_w_y, hi_w_y):
    dataframe = pd.DataFrame(dataframe)
    
    df0 = dataframe.loc[(dataframe['Label'] == 0) & (dataframe['w_x']>=int(lo_w_x)) & (dataframe['w_x'] <= int(hi_w_x)) & (dataframe['w_y'] >= int(lo_w_y)) & (dataframe['w_y'] <= int(hi_w_y))] #M500
    df1 = dataframe.loc[(dataframe['Label'] == 1) & (dataframe['w_x']>=int(lo_w_x)) & (dataframe['w_x'] <= int(hi_w_x)) & (dataframe['w_y'] >= int(lo_w_y)) & (dataframe['w_y'] <= int(hi_w_y))] #M1000
    df2 = dataframe.loc[(dataframe['Label'] == 2) & (dataframe['w_x']>=int(lo_w_x)) & (dataframe['w_x'] <= int(hi_w_x)) & (dataframe['w_y'] >= int(lo_w_y)) & (dataframe['w_y'] <= int(hi_w_y))] #M1500
    df3 = dataframe.loc[(dataframe['Label'] == 4) & (dataframe['w_x']>=int(lo_w_x)) & (dataframe['w_x'] <= int(hi_w_x)) & (dataframe['w_y'] >= int(lo_w_y)) & (dataframe['w_y'] <= int(hi_w_y))] #SM(TTZ+WZ)
    
    cluster_name = pd.concat([df0, df1, df2, df3])

    array_name=[]
    total500 = 0
    total1000 = 0
    total1500 = 0
    totalSM = 0

    for i in np.arange(int(lo_w_x), int(hi_w_x+1)):
        for j in np.arange(int(lo_w_y), int(hi_w_y+1)):
            num500 = ((cluster_name['w_x'] == i) & (cluster_name['w_y'] == j) & (cluster_name['Label'] == 0)).sum() 
            num1000 = ((cluster_name['w_x'] == i) & (cluster_name['w_y'] == j) & (cluster_name['Label'] == 1)).sum()
            num1500 = ((cluster_name['w_x'] == i) & (cluster_name['w_y'] == j) & (cluster_name['Label'] == 2)).sum()
            numSM = ((cluster_name['w_x'] == i) & (cluster_name['w_y'] == j) & (cluster_name['Label'] == 4)).sum()
            
            total500 += num500
            total1000 += num1000
            total1500 += num1500
            totalSM += numSM
            
            array_name.append([i, j, num500, num1000, num1500, numSM])
            
            #for loops end here
            
    array_name = pd.DataFrame(array_name, columns = [f'w_x_{ClusterNum}', f'w_y_{ClusterNum}', f'num500_{ClusterNum}', f'num1000_{ClusterNum}', f'num1500_{ClusterNum}', f'numSM_{ClusterNum}'])
    
    array_name[f'clus{ClusterNum}Score500'] = array_name.apply(lambda row:((row[f'num500_{ClusterNum}'] - row[f'num1000_{ClusterNum}'] - row[f'num1500_{ClusterNum}'] - row[f'numSM_{ClusterNum}'])) / (row[f'num500_{ClusterNum}'] + row[f'num1000_{ClusterNum}'] + row[f'num1500_{ClusterNum}'] + row[f'numSM_{ClusterNum}']), axis=1)
    
    array_name[f'clus{ClusterNum}Score1000'] = array_name.apply(lambda row:((row[f'num1000_{ClusterNum}'] - row[f'num500_{ClusterNum}'] - row[f'num1500_{ClusterNum}'] - row[f'numSM_{ClusterNum}'])) / (row[f'num500_{ClusterNum}'] + row[f'num1000_{ClusterNum}'] + row[f'num1500_{ClusterNum}'] + row[f'numSM_{ClusterNum}']), axis=1)
    
    array_name[f'clus{ClusterNum}Score1500'] = array_name.apply(lambda row:((row[f'num1500_{ClusterNum}'] - row[f'num500_{ClusterNum}'] - row[f'num1000_{ClusterNum}'] - row[f'numSM_{ClusterNum}'])) / (row[f'num500_{ClusterNum}'] + row[f'num1000_{ClusterNum}'] + row[f'num1500_{ClusterNum}'] + row[f'numSM_{ClusterNum}']), axis=1)

    array_name[f'clus{ClusterNum}ScoreSM'] = array_name.apply(lambda row:((row[f'numSM_{ClusterNum}'] - row[f'num500_{ClusterNum}'] - row[f'num1000_{ClusterNum}'] - row[f'num1500_{ClusterNum}'])) / (row[f'num500_{ClusterNum}'] + row[f'num1000_{ClusterNum}'] + row[f'num1500_{ClusterNum}'] + row[f'numSM_{ClusterNum}']), axis=1)

    nonEmptyNeurons = ((array_name[f'num500_{ClusterNum}'] + array_name[f'num1000_{ClusterNum}'] + array_name[f'num1500_{ClusterNum}'] + array_name[f'numSM_{ClusterNum}'] > 0)).sum() 

    sepScore500 = array_name[f'clus{ClusterNum}Score500'].sum()
    sepScore500 = sepScore500 / nonEmptyNeurons

    sepScore1000 = array_name[f'clus{ClusterNum}Score1000'].sum()
    sepScore1000 = sepScore1000 / nonEmptyNeurons
    
    sepScore1500 = array_name[f'clus{ClusterNum}Score1500'].sum()
    sepScore1500 = sepScore1500 / nonEmptyNeurons

    sepScoreSM = array_name[f'clus{ClusterNum}ScoreSM'].sum()
    sepScoreSM = sepScoreSM / nonEmptyNeurons


    
    #plt.figure(figsize=(8.27, 11.69))
    #txt0 = f"--------------------------------Cluster {ClusterNum} Summary---------------------------------" '\n' " " " \n" f"The cluster{ClusterNum} lies in the region: " ' \n' f"x = {lo_w_x} to {hi_w_x}" '\n' f"y = {lo_w_y} to {hi_w_y}" '\n' "extreme neurons included" '\n' " " '\n' f"The total neurons in cluster {ClusterNum} = {(int(hi_w_x+1) - int(lo_w_x)) * (int(hi_w_y+1) - int(lo_w_y))}, NonEmptyNeurons in the cluster {ClusterNum}= {nonEmptyNeurons}"  '\n' " "  '\n' f"Cluster {ClusterNum} SepScore500 = {sepScore500:.3f}" '\n' f"Cluster {ClusterNum} SepScore1000 = {sepScore1000:.3f}" '\n' f"Cluster {ClusterNum} SepScore1500 = {sepScore1500:.3f}" '\n' f"Cluster {ClusterNum} SepScoreBkg = {sepScoreBkg:.3f}" '\n' f"Cluster {ClusterNum} AOF = {TotalAOF:.3f}" '\n' " " '\n' f"Total number of M500 points in the cluster = {total500}" '\n' f"Total number of M1000 points in the cluster = {total1000}" '\n' f"Total number of M1500 points in the cluster = {total1500}" '\n' " " '\n'
    #plt.text(0.5, 0.90, txt0, ha='center', va='top', wrap=True, fontsize=12)
    #plt.axis('off')
    #plt.savefig(pp, format='pdf')
   
    
    return [round(sepScore500,3), round(sepScore1000,3), round(sepScore1500,3), round(sepScoreSM,3)]

#------------------------------------------------------------------------------------------------------------------------
'''


#'''
#------------------------Cluster_score function-----------------------------------------------------
#def ClusterScore_sd(dataframe, lox,hix,loy,hiy):
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

    #print()
    #print('arr')
    #print(arr)
    
    arr['Score500'] =arr.apply(lambda row: (row['num500']-row['num1000']-row['num1500']-row['numSM'])/(row['num500']+row['num1000']+row['num1500']+row['numSM']) if (row['num500']+row['num1000']+row['num1500']+row['numSM'])>0 else -1, axis=1)
    arr['Score1000']=arr.apply(lambda row: (row['num1000']-row['num500']-row['num1500']-row['numSM'])/(row['num500']+row['num1000']+row['num1500']+row['numSM']) if (row['num500']+row['num1000']+row['num1500']+row['numSM'])>0 else -1, axis=1)
    arr['Score1500']=arr.apply(lambda row: (row['num1500']-row['num500']-row['num1000']-row['numSM'])/(row['num500']+row['num1000']+row['num1500']+row['numSM']) if (row['num500']+row['num1000']+row['num1500']+row['numSM'])>0 else -1, axis=1)
    arr['ScoreSM']=arr.apply(lambda row: (row['numSM']-row['num500']-row['num1000']-row['num1500'])/(row['num500']+row['num1000']+row['num1500']+row['numSM']) if (row['num500']+row['num1000']+row['num1500']+row['numSM'])>0 else -1, axis=1)
    arr['isFilled'] = arr.apply(lambda row: 1 if (row['num500']+row['num1000']+row['num1500']+row['numSM']) >0  else 0, axis=1)

    #print()
    #print('arr with score:')
    #print(arr)
    
    nonEmptyNeurons = ((arr['num500']+arr['num1000']+arr['num1500']+arr['numSM'] >0)).sum()

    #print()
    #print(f'number of Non-empty neurons = {nonEmptyNeurons}')

    sepScore500 = -1
    sepScore1000 = -1
    sepScore1500 = -1
    sepScoreSM = -1
    if( nonEmptyNeurons > 0):
        filled = arr[arr['isFilled'] == 1]
        #print()
        #print('filled:')
        #print(filled)
        sepScore500  = filled['Score500'].sum()
        sepScore500  = sepScore500 / nonEmptyNeurons
        sepScore1000 = filled['Score1000'].sum()
        #print(f'sepScore1000 = {sepScore1000}')
        sepScore1000 = sepScore1000 / nonEmptyNeurons
        #print(f'sepScore1000/nen = {sepScore1000}')
        sepScore1500 = filled['Score1500'].sum()
        sepScore1500 = sepScore1500 / nonEmptyNeurons
        sepScoreSM   = filled['ScoreSM'].sum()
        sepScoreSM  = sepScoreSM / nonEmptyNeurons
    
    #print(f'(SS500, SS1000, SS1500, SSSM) = {sepScore500}, {sepScore1000}, {sepScore1500}, {sepScoreSM}')
    return [round(sepScore500,3), round(sepScore1000,3), round(sepScore1500,3), round(sepScoreSM,3)]   

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#'''



#-----------------------------------------Cases Evalutation function------------------------------------------
def caseFunct(case_num, w_x, w_y, unnorm_case_arr, case_arr, case_lab_arr, case_sum, use, pdf, title, som_arr, rangeA, rangeB, ranAclus_arr, ranBclus_arr, scores_ranA_arr, scores_ranB_arr, test_df, som_dim0, som_dim1):
    w_x = np.array(w_x)
    w_y = np.array(w_y)
    w_xy = np.column_stack((w_x, w_y))

    case_df = np.column_stack((case_arr, w_x, w_y, case_lab_arr))
    case_df = pd.DataFrame(case_df)
    case_df.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'w_x', 'w_y', 'Label']
    
    case_sum=[]
    SepScore(case_sum, case_df, use, som_dim0, som_dim1) #This is for the whole case. It is the score when points af that case are distributed on the testSOM grid. It is NOT regional. 
    #plt.savefig(pdf, format = 'pdf')

    plt.figure(figsize=(8.27,11.69))
    text1 = f"Case {case_num}"
    plt.text(0.5, 0.90, text1, ha = 'center', va = 'top', wrap = True, fontsize=12)
    plt.axis('off')
    plt.savefig(pdf, format = 'pdf')

    color_list = {0: 'teal', 1: 'blue', 2: 'red', 3: 'purple', 4: 'black', 5: 'orange', 6: 'green', 7: 'magenta'}
    label_list = {0: 'M500', 1: 'M1000', 2: 'M1500', 3: 'M750', 4: 'SM', 5: 'M2500', 6: 'M600', 7: 'M800'}
    #marker_list = {0: 'o', 1: 'o', 2: 'o', 3: 'o', 4: 'x', 5: 'o'}
    marker_list = {0: 'o', 1: '^', 2: 's', 3: 'P', 4: 'x', 5: '*', 6: '<', 7: 'D'}

    
    plt.figure(figsize=(20,20))
    #plt.title(title, fontsize=12)
    plt.pcolor(som_arr, cmap = 'bone_r', alpha = 0.0)

    for j in np.unique(case_lab_arr):
        idx_label = case_lab_arr == j
        plt.scatter(w_x[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, w_y[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, s=300, c = 'xkcd:faded purple', label = label_list[j], marker = 'P')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    #ax.xaxis.set_minor_locator(MultipleLocator(2))
    #ax.yaxis.set_minor_locator(MultipleLocator(2))
    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1, alpha = 0.9)
    plt.tick_params(axis='both', labelsize=25)
    #plt.title('Training_SOM_v1', fontsize=15)
    #plt.legend(loc = 'upper right', fontsize=30)
    plt.xlabel('Neuron X', fontsize=35)
    plt.ylabel('Neuron Y', fontsize=35)
    plt.savefig(pdf, bbox_inches = 'tight', pad_inches=0, format = 'pdf')
    
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
        #scores_ranA_arr[m] = ClusterScore(test_df, m, ranAclus_arr[m][0], ranAclus_arr[m][1], ranAclus_arr[m][2], ranAclus_arr[m][3]) #This is the regional sepscore (non-absolute)
        scores_ranA_arr[m] = ClusterScore(test_df, ranAclus_arr[m][0], ranAclus_arr[m][1], ranAclus_arr[m][2], ranAclus_arr[m][3]) #This is the regional sepscore (non-absolute)
        #plt.savefig(pdf, format = 'pdf')

    #print()
    #print('scores_ranA_arr')
    #print(scores_ranA_arr)
    
    scores_ranA_arr = np.array(scores_ranA_arr)
    meanSS500A = np.mean([row[0] for row in scores_ranA_arr])
    meanSS1000A = np.mean([row[1] for row in scores_ranA_arr])
    meanSS1500A = np.mean([row[2] for row in scores_ranA_arr])
    meanSSSMA = np.mean([row[3] for row in scores_ranA_arr])

    medSS500A = np.median([row[0] for row in scores_ranA_arr])
    medSS1000A = np.median([row[1] for row in scores_ranA_arr])
    medSS1500A = np.median([row[2] for row in scores_ranA_arr])
    medSSSMA = np.median([row[3] for row in scores_ranA_arr])

    #f'SepScore500 ({medSS500A:.3f})'
    
    plt.figure(figsize=(8,8))
    plt.hist(scores_ranA_arr[: , 0], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{500}}} $" + f'({medSS500A:.3f})', linewidth = 3, color = 'xkcd:pinkish purple', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_arr[: , 1], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1000}}} $" + f'({medSS1000A:.3f})', linewidth = 3, color = 'xkcd:tangerine', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_arr[: , 2], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1500}}} $" + f'({medSS1500A:.3f})', linewidth = 3, color = 'xkcd:murky green', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_arr[: , 3], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{SM}}} $" + f'({medSSSMA:.3f})', linewidth = 3, color = 'black', density = False, log = False, alpha = 0.3)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc= 'upper right', fontsize=15)
    plt.xlabel(f'Regional SepScore ({2*rangeA+1} x {2*rangeA+1})', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    #plt.title(f'Case {case_num}: Hist for scores_+-{rangeA}_region around points', fontsize=12)
    plt.tick_params(axis='both', which = 'major', labelsize=15)
    #plt.savefig(pdf, format = 'pdf')
    plt.savefig(pdf, bbox_inches = 'tight', pad_inches=0, format = 'pdf')

    #Case_lab, SS500, SS1000, SS1500, SSSM
    case_lab_arr2 = np.array(case_lab_arr)[:, np.newaxis]
    scores_labels_ranA = np.hstack([case_lab_arr2, scores_ranA_arr])
    scores_labels_ranA = pd.DataFrame(scores_labels_ranA)
    scores_labels_ranA.columns = ['Label', 'SS500', 'SS1000', 'SS1500', 'SSSM']
    
    #Cutting on the sepScoreSM < 0.6 for rangeA (+-2 around the point)
    scores_ranA_SMcut = [row for row in scores_ranA_arr if row[-1] < 0.6]
    scores_ranA_SMcut = np.array(scores_ranA_SMcut)
    meanSS500Acut = np.mean([row[0] for row in scores_ranA_SMcut])
    meanSS1000Acut = np.mean([row[1] for row in scores_ranA_SMcut])
    meanSS1500Acut = np.mean([row[2] for row in scores_ranA_SMcut])
    meanSSSMAcut = np.mean([row[3] for row in scores_ranA_SMcut])

    medSS500Acut = np.median([row[0] for row in scores_ranA_SMcut])
    medSS1000Acut = np.median([row[1] for row in scores_ranA_SMcut])
    medSS1500Acut = np.median([row[2] for row in scores_ranA_SMcut])
    medSSSMAcut = np.median([row[3] for row in scores_ranA_SMcut])
    
    #Picking only those rows for which SSBkg <0.6 from scores_labels_ranA into scores_labels_ranA_Bkgcut
    scores_labels_ranA_SMcut = scores_labels_ranA[scores_labels_ranA['SSSM'] < 0.6]
    #print()
    #print('score ranA Bafter 0.6 cut')    
    #print(scores_labels_ranA_SMcut)
    
    #Text page for range of +-2 around point
    plt.figure(figsize=(8.27,11.69))
    textA = f"Case {case_num}: cluster of +- {rangeA} around the point" '\n' " " '\n' "Array of label and scores BEFORE ssSM < 0.6 cut: [label, ss500, ss1000, ss1500, ssSM]:" '\n' f'{scores_labels_ranA}' '\n' " " '\n' "Array of label and scores AFTER ssSM < 0.6 cut: [label, ss500, ss1000, ss1500, ssSM]:" '\n' f'{scores_labels_ranA_SMcut}' '\n' " " '\n' f'Array of scores BEFORE ssSM < 0.6 cut: {len(scores_ranA_arr)}' '\n' f'Array of scores AFTER ssSM < 0.6 cut: {len(scores_ranA_SMcut)}' '\n'
    plt.text(0.5, 0.90, textA, ha = 'center', va = 'top', wrap = True, fontsize=12)
    plt.axis('off')
    plt.savefig(pdf, format = 'pdf')

    plt.figure(figsize=(8,8))
    plt.hist(scores_ranA_SMcut[: , 0], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{500}}} $" + f'({medSS500Acut:.3f})', linewidth = 3, color = 'xkcd:pinkish purple', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_SMcut[: , 1], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1000}}} $" + f'({medSS1000Acut:.3f})', linewidth = 3, color = 'xkcd:tangerine', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_SMcut[: , 2], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1500}}} $" + f'({medSS1500Acut:.3f})', linewidth = 3, color = 'xkcd:murky green', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranA_SMcut[: , 3], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label =r"${\mathrm{SepScore_{SM}}} $" + f'({medSSSMAcut:.3f})', linewidth = 3, color = 'black', density = False, log = False, alpha = 0.3)
    #plt.legend(loc= 'upper right')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(f'Regional SepScore ({2*rangeA+1} x {2*rangeA+1})', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    #plt.title(f'Case {case_num}: Hist for scores_+-{rangeA}_region around points - sepScoreSM < 0.6', fontsize=12)
    plt.legend(loc= 'upper right', fontsize=15)
    plt.tick_params(axis='both', which = 'major', labelsize=15)    
    #plt.legend(loc= 'upper right')
    #plt.savefig(pdf, format = 'pdf')
    plt.savefig(pdf, bbox_inches = 'tight', pad_inches=0, format = 'pdf')

    #Array: [[....(input vars).., w_x, w_y, Label, SepScore500, SepScore1000, SepScore1500, SepScoreSM]]
    #The input vars, w_x, w_y, Label come from case_df and the SepScores come from scores_ranA_arr
    #case_npArr = np.array(case_df)
    #VarsScore = np.column_stack([case_npArr, scores_ranA_arr])
    VarsScore_ranA = np.column_stack([unnorm_case_arr, w_x, w_y, case_lab_arr, scores_ranA_arr])
    VarsScore_ranA = pd.DataFrame(VarsScore_ranA)
    VarsScore_ranA.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'w_x', 'w_y', 'Label', 'SS500', 'SS1000', 'SS1500', 'SSSM']

    #print()
    #print('VarsScore_ranA')
    #print(VarsScore)

    #Cut on SSM score<0.6, and plot a hist of all those rows which satisfy the cut.
    VarsScore_ranA_cut = VarsScore_ranA[VarsScore_ranA['SSSM'] < 0.6]
    #print()
    #print('VarsScore_cut_ranA')
    #print(VarsScore_cut)
    plotnamesA = ['trilepM']
    xlabA = [ r"$m_{\ell\ell\ell}$"]
    mybinsA = np.arange(0,4000,160)
    mybinsA1 = np.arange(0,4000, 40)
    PlotVars(VarsScore_ranA_cut, mybinsA, mybinsA1, plotnamesA, xlabA, pdf, rangeA, case_num)
    
    #RangeB -----------------------
    ranBclus_arr = [[0,0,0,0] for _ in range(len(case_df))]
    for j in range(0,len(case_df)):
        ranBclus_arr[j][0] = case_df.iloc[j,8] - rangeB #w_x_lo
        ranBclus_arr[j][1] = case_df.iloc[j,8] + rangeB #w_x_hi
        ranBclus_arr[j][2] = case_df.iloc[j,9] - rangeB #w_y_lo
        ranBclus_arr[j][3] = case_df.iloc[j,9] + rangeB #w_y_hi

    for k in range(0, len(case_df)):
        for l in range(0,4):
            if(ranBclus_arr[k][l] < 0):
                ranBclus_arr[k][l] = 0
            if(ranBclus_arr[k][l] > 100):
                ranBclus_arr[k][l] = 100

    ranBclus_arr = np.array(ranBclus_arr)
    scores_ranB_arr = [[0,0,0,0] for _ in range(len(case_df))]
    for n in range(len(case_df)):
        #scores_ranB_arr[m] = ClusterScore(test_df, m, ranBclus_arr[m][0], ranBclus_arr[m][1], ranBclus_arr[m][2], ranBclus_arr[m][3]) #This is the regional sepscore (non-absolute)
        scores_ranB_arr[n] = ClusterScore(test_df, ranBclus_arr[n][0], ranBclus_arr[n][1], ranBclus_arr[n][2], ranBclus_arr[n][3]) #This is the regional sepscore (non-absolute)
        #plt.savefig(pdf, format = 'pdf')

    scores_ranB_arr = np.array(scores_ranB_arr)
    meanSS500B = np.mean([row[0] for row in scores_ranB_arr])
    meanSS1000B = np.mean([row[1] for row in scores_ranB_arr])
    meanSS1500B = np.mean([row[2] for row in scores_ranB_arr])
    meanSSSMB = np.mean([row[3] for row in scores_ranB_arr])

    medSS500B = np.median([row[0] for row in scores_ranB_arr])
    medSS1000B = np.median([row[1] for row in scores_ranB_arr])
    medSS1500B = np.median([row[2] for row in scores_ranB_arr])
    medSSSMB = np.median([row[3] for row in scores_ranB_arr])

    
    plt.figure(figsize=(8,8))
    plt.hist(scores_ranB_arr[: , 0], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{500}}} $" + f'({medSS500B:.3f})', linewidth = 3, color = 'xkcd:pinkish purple', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranB_arr[: , 1], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1000}}} $" + f'({medSS1000B:.3f})', linewidth = 3, color = 'xkcd:tangerine', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranB_arr[: , 2], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1500}}} $" + f'({medSS1500B:.3f})', linewidth = 3, color = 'xkcd:murky green', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranB_arr[: , 3], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label =r"${\mathrm{SepScore_{SM}}} $" + f'({medSSSMB:.3f})', linewidth = 3, color = 'black', density = False, log = False, alpha = 0.3)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc= 'upper right', fontsize=15)
    plt.xlabel(f'Regional SepScore ({2*rangeB+1} x {2*rangeB+1})', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    #plt.title(f'Case {case_num}: Hist for scores_+-{rangeA}_region around points', fontsize=12)
    plt.tick_params(axis='both', which = 'major', labelsize=15)
    #plt.savefig(pdf, format = 'pdf')
    plt.savefig(pdf, bbox_inches = 'tight', pad_inches=0, format = 'pdf')

    #Case_lab, SS500, SS1000, SS1500, SSSM
    case_lab_arr2 = np.array(case_lab_arr)[:, np.newaxis]
    scores_labels_ranB = np.hstack([case_lab_arr2, scores_ranB_arr])
    scores_labels_ranB = pd.DataFrame(scores_labels_ranB)
    scores_labels_ranB.columns = ['Label', 'SS500', 'SS1000', 'SS1500', 'SSSM']
    
    #Cutting on the sepScoreSM < 0.6 for rangeB (+-2 around the point)
    scores_ranB_SMcut = [row for row in scores_ranB_arr if row[-1] < 0.6]
    scores_ranB_SMcut = np.array(scores_ranB_SMcut)
    meanSS500Bcut = np.mean([row[0] for row in scores_ranB_SMcut])
    meanSS1000Bcut = np.mean([row[1] for row in scores_ranB_SMcut])
    meanSS1500Bcut = np.mean([row[2] for row in scores_ranB_SMcut])
    meanSSSMBcut = np.mean([row[3] for row in scores_ranB_SMcut])

    medSS500Bcut = np.median([row[0] for row in scores_ranB_SMcut])
    medSS1000Bcut = np.median([row[1] for row in scores_ranB_SMcut])
    medSS1500Bcut = np.median([row[2] for row in scores_ranB_SMcut])
    medSSSMBcut = np.median([row[3] for row in scores_ranB_SMcut])
    
    #Picking only those rows for which SSBkg <0.6 from scores_labels_ranB into scores_labels_ranB_Bkgcut
    scores_labels_ranB_SMcut = scores_labels_ranB[scores_labels_ranB['SSSM'] < 0.6]
    #print()
    #print('score ranB Bafter 0.6 cut')
    #print(scores_labels_ranB_SMcut)
    
    #Text page for range of +-2 around point
    plt.figure(figsize=(8.27,11.69))
    textB = f"Case {case_num}: cluster of +- {rangeB} around the point" '\n' " " '\n' "Array of label and scores BEFORE ssSM < 0.6 cut: [label, ss500, ss1000, ss1500, ssSM]:" '\n' f'{scores_labels_ranB}' '\n' " " '\n' "Array of label and scores AFTER ssSM < 0.6 cut: [label, ss500, ss1000, ss1500, ssSM]:" '\n' f'{scores_labels_ranB_SMcut}' '\n' " " '\n' f'Array of scores BEFORE ssSM < 0.6 cut: {len(scores_ranB_arr)}' '\n' f'Array of scores AFTER ssSM < 0.6 cut: {len(scores_ranB_SMcut)}'
    plt.text(0.5, 0.90, textB, ha = 'center', va = 'top', wrap = True, fontsize=12)
    plt.axis('off')
    plt.savefig(pdf, format = 'pdf')

    plt.figure(figsize=(8,8))
    plt.hist(scores_ranB_SMcut[: , 0], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{500}}} $" + f'({medSS500Bcut:.3f})', linewidth = 3, color = 'xkcd:pinkish purple', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranB_SMcut[: , 1], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1000}}} $" + f'({medSS1000Bcut:.3f})', linewidth = 3, color = 'xkcd:tangerine', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranB_SMcut[: , 2], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{1500}}} $" + f'({medSS1500Bcut:.3f})', linewidth = 3, color = 'xkcd:murky green', density = False, log = False, alpha = 0.5)
    plt.hist(scores_ranB_SMcut[: , 3], bins = np.arange(-1.0,1.4,0.2), histtype = 'step', label = r"${\mathrm{SepScore_{SM}}} $" + f'({medSSSMBcut:.3f})', linewidth = 3, color = 'black', density = False, log = False, alpha = 0.3)
    #plt.legend(loc= 'upper right')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(f'Regional SepScore ({2*rangeB+1} x {2*rangeB+1})', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    #plt.title(f'Case {case_num}: Hist for scores_+-{rangeA}_region around points - sepScoreSM < 0.6', fontsize=12)
    plt.legend(loc= 'upper right', fontsize=15)
    plt.tick_params(axis='both', which = 'major', labelsize=15)    
    #plt.legend(loc= 'upper right')
    #plt.savefig(pdf, format = 'pdf')
    plt.savefig(pdf, bbox_inches = 'tight', pad_inches=0, format = 'pdf')

    #Array: [[....(input vars).., w_x, w_y, Label, SepScore500, SepScore1000, SepScore1500, SepScoreSM]]
    #The input vars, w_x, w_y, Label come from case_df and the SepScores come from scores_ranB_arr
    #case_npArr = np.array(case_df)
    #VarsScore = np.column_stack([case_npArr, scores_ranB_arr])
    VarsScore_ranB = np.column_stack([unnorm_case_arr, w_x, w_y, case_lab_arr, scores_ranB_arr])
    VarsScore_ranB = pd.DataFrame(VarsScore_ranB)
    VarsScore_ranB.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'w_x', 'w_y', 'Label', 'SS500', 'SS1000', 'SS1500', 'SSSM']

    #print()
    #print('VarsScore_ranB')
    #print(VarsScore)

    #Cut on SSM score<0.6, and plot a hist of all those rows which satisfy the cut.
    VarsScore_ranB_cut = VarsScore_ranB[VarsScore_ranB['SSSM'] < 0.6]
    #print()
    #print('VarsScore_cut_ranB')
    #print(VarsScore_cut)
    plotnamesB = ['trilepM']
    xlabB = [ r"$m_{\ell\ell\ell}$"]
    mybinsB = np.arange(0,4000,160)
    mybinsB1 = np.arange(0,4000, 40)
    PlotVars(VarsScore_ranB_cut, mybinsB, mybinsB1, plotnamesB, xlabB, pdf, rangeB, case_num)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------




#------------------FOR ROC-----------------
def caseFunct_mod(case_num, w_x, w_y, case_arr, case_lab_arr, pdf, title, som_arr, rangeA, ranAclus_arr, scores_ranA_arr, test_df, number):    
    w_x = np.array(w_x)
    w_y = np.array(w_y)
    w_xy = np.column_stack((w_x, w_y))

    case_df = np.column_stack((case_arr, w_x, w_y, case_lab_arr))
    case_df = pd.DataFrame(case_df)
    case_df.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'w_x', 'w_y', 'Label']

    case_sum=[]

    plt.figure(figsize=(20,20))
    plt.title(title, fontsize=12)
    plt.pcolor(som_arr, cmap = 'bone_r', alpha = 0.0)

    color_list = {0: 'teal', 1: 'blue', 2: 'red', 3: 'purple', 4: 'black', 5: 'orange'}
    label_list = {0: 'M500', 1: 'M1000', 2: 'M1500', 3: 'M750', 4: 'SM', 5: 'M2500'}
    #marker_list = {0: 'o', 1: 'o', 2: 'o', 3: 'o', 4: 'x', 5: 'o'}
    marker_list = {0: 'o', 1: '^', 2: 's', 3: 'P', 4: 'x', 5: '*'}
    
    for i in np.unique(case_lab_arr):
        idx_label = case_lab_arr == i
        plt.scatter(w_x[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, w_y[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, s=300, c = color_list[i], label = label_list[i], marker = marker_list[i], alpha = 0.4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    plt.xlabel('Neuron X', fontsize=30)
    plt.ylabel('Neuron Y', fontsize=30)
    plt.legend(loc = 'upper right', fontsize=30)
    plt.tick_params(axis='both', which = 'major', labelsize=25)
    plt.savefig(pdf, format = 'pdf')

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
            if(ranAclus_arr[k][l] > number):
                ranAclus_arr[k][l] = number
    
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

# ---------------------------------------------------
# takes a dataframe to plot the ROC curves
#
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

    plt.figure(figsize=(8,8))
    plt.plot(fpr500, tpr500, label=r'$\mathrm{SepScore^{reg}_{500}}$' + f'(AUC={auc500:.3f})', color='xkcd:pinkish purple', lw=2)
    plt.plot(fpr1000, tpr1000, label=r'$\mathrm{SepScore^{reg}_{1000}}$' + f'(AUC={auc1000:.3f})', color='xkcd:tangerine', lw=2)
    plt.plot(fpr1500, tpr1500, label=r'$\mathrm{SepScore^{reg}_{1500}}$' + f'(AUC={auc1500:.3f})', color='xkcd:murky green', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Diagonal 'random guess' line
    plt.xlabel('Background Efficiency')
    plt.ylabel('Signal Efficiency')
    plt.title('ROC from SOM')
    plt.legend(loc='lower right')
    plt.savefig(pdf, format='pdf')

    plt.figure(figsize=(8,8))
    plt.hist(M500_S500,bins=np.arange(-1.0,1.4,0.1),alpha=0.9,linewidth=3,color='xkcd:pinkish purple',density=True,log=False,histtype='step',label='M500')
    plt.hist(Mothers_S500,bins=np.arange(-1.0,1.4,0.1),alpha=0.4,linewidth=3,color='black',density=True,log=False,histtype='step',label='Others')
    plt.legend(loc= 'upper right', fontsize=15)
    plt.xlabel('Separation Score 500', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    plt.savefig(pdf, format='pdf')

    plt.figure(figsize=(8,8))
    plt.hist(M1000_S1000,bins=np.arange(-1.0,1.4,0.1),alpha=0.9,linewidth=3,color='xkcd:tangerine',density=True,log=False,histtype='step',label='M1000')
    plt.hist(Mothers_S1000,bins=np.arange(-1.0,1.4,0.1),alpha=0.4,linewidth=3,color='black',density=True,log=False,histtype='step',label='Others')
    plt.legend(loc= 'upper right', fontsize=15)
    plt.xlabel('Separation Score 1000', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    plt.savefig(pdf, format='pdf')

    plt.figure(figsize=(8,8))
    plt.hist(M1500_S1500,bins=np.arange(-1.0,1.4,0.1),alpha=0.9,linewidth=3,color='xkcd:murky green',density=True,log=False,histtype='step',label='M1500')
    plt.hist(Mothers_S1000,bins=np.arange(-1.0,1.4,0.1),alpha=0.4,linewidth=3,color='black',density=True,log=False,histtype='step',label='Others')
    plt.legend(loc= 'upper right', fontsize=15)
    plt.xlabel('Separation Score 1500', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    plt.savefig(pdf, format='pdf')
