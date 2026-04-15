#!/usr/bin/env python
# coding: utf-8


#Import the necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,auc, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys
import warnings
import time
warnings.filterwarnings('ignore')

from DNNfunctions import *

# ------------------------------------------------------------
# Setup the names of the output files
# 
# ------------------------------------------------------------

foldername = "outputs/"
modelname = foldername+"mc_model.h5"

outputname = foldername+"/plots_cases.pdf"
textfilename = foldername+"/sum_cases.txt"
    
pp = PdfPages(outputname)


# ------------------------------------------------------------
# Define the names of the input variables
# from the text files.
# ------------------------------------------------------------

col_names = ['LT_3L', 'HT_3L', 'njets', 'j0pT', 'MET_3L', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'Mos1_3L', 'Mos2_3L', 'trilepM', 'mT0', 'mT1', 'mT2', 'highmT_3L', '3lmT_3L', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'pT0_3L', 'pT1_3L', 'pT2_3L' , 'pT_tau_3L']

train_vars = ['LT_3L', 'HT_3L', 'Mos1_3L', 'Mos2_3L', 'trilepM', '3lmT_3L', 'highmT_3L','pT0_3L', 'j0pT']
cols = [col_names.index(var) for var in train_vars]


# ------------------------------------------------------------
# Load the dataframes
# 
# ------------------------------------------------------------

text_path = '../inputs'

VLL_m500_cases = pd.read_csv(text_path + '/M500/M500_500Ev_cases.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000_cases = pd.read_csv(text_path + '/M1000/M1000_500Ev_cases.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1500_cases = pd.read_csv(text_path + '/M1500/M1500_500Ev_cases.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m750_cases = pd.read_csv(text_path + '/M750/M750_500Ev_cases.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m2500_cases = pd.read_csv(text_path + '/M2500/M2500_500Ev_cases.txt',sep=' ',index_col=None, usecols=cols,names=col_names)

WZ_cases = pd.read_csv(text_path + '/SM/WZ_500Ev_cases.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
TTZ_cases = pd.read_csv(text_path + '/SM/TTZ_500Ev_cases.txt',sep=' ',index_col=None, usecols=cols,names=col_names)


#These are the variables used during training
train_vars = ['LT_3L', 'HT_3L', 'Mos1_3L', 'Mos2_3L', 'trilepM', '3lmT_3L', 'highmT_3L', 'pT0j0pT'] #Training variables


alldfs = [VLL_m500_cases, VLL_m1000_cases, VLL_m1500_cases, VLL_m750_cases, VLL_m2500_cases, WZ_cases , TTZ_cases]
for df in alldfs:
    df['pT0j0pT'] = df['pT0_3L'] + df['j0pT']



#=====================================

#Now we must declare arrays to keep the min, max, mean and std values that we got from train_DNN.py


maxVals = np.array([[5318.69, 6431.1, 5303.18, 2365.03, 5483.48, 4579.97, 4123.02, 5614.6]])
minVals = np.array([[57.2085, 0.0, 28.3349, 0.0, 41.194, 0.0, 10.4011, 54.4745]])

meanVals = np.array([[897.7713998868422, 1248.6450995657895, 713.5687986157893, 367.6396013010527, 974.0936867763157, 527.0796259861737, 595.1327105131579, 983.8834515631579]])
stdVals = np.array([[656.6223238386779, 866.0036089099726, 596.5449094829836, 349.27500195562743, 747.120547207153, 526.2393190428262, 504.81672325614255, 706.4334590124485]])

#Using the standrad normalization
arg1 = meanVals; arg2 = stdVals
#=====================================

### Loading the testing datasets for the individual cases
### Using random sampling to get the specific cases

testdata_names = ["case1: 10M1000", "case2: 10M2500", "case3: 10M500_10Bkg", "case4: 10M750_5Bkg"]

testdata_dfs = [
    pd.concat([VLL_m1000_cases.sample(10)]),
    pd.concat([VLL_m2500_cases.sample(10)]),
    pd.concat([VLL_m500_cases.sample(10), WZ_cases.sample(5), TTZ_cases.sample(5)]),
    pd.concat([VLL_m750_cases.sample(10), WZ_cases.sample(2), TTZ_cases.sample(3)])
]



### Getting the NN model
mymodel = tf.keras.models.load_model(modelname)
mymodel.load_weights(modelname)


# ------------------------------------------------------------
# Now we do the prediction for the individual cases
# plot the neuron scores with and without a cut on bkg_score.
# ------------------------------------------------------------
txtfile = open(textfilename, "w")
for i in range(4):
    
    testdata = testdata_dfs[i][train_vars]

    #Normalising the testing dataset
    norm_testdata = normalise_df(testdata, arg1, arg2)

    # Predicting the NNScores
    NNScores = mymodel.predict(norm_testdata)

    ### Plotting the NN score without and with bkg cut:
    plot_one_process_score(NNScores, testdata_names[i], pp, with_bkg_score_cut=False)
    plot_one_process_score(NNScores, testdata_names[i], pp, with_bkg_score_cut=True)


    ### Median neuron scores with a bkg cut
    median_score = np.median(NNScores[NNScores[:,3]<0.8],axis=0)

    # ------------------------------------------------------------
    # Print some information and write to text file
    # 
    # ------------------------------------------------------------
    
    
    print("Test Data Composition: ", testdata_names[i])
    print("Average Neuron scores:", median_score)
    print("*"*30+"\n\n")

    txtfile.write("Test Data Composition: "+testdata_names[i]+"\n")
    txtfile.write("Avg Neuron scores: "+str(median_score)+"\n")
    txtfile.write("*"*30+"\n\n")


print(f'Output is saved as {outputname} and {textfilename}.')

pp.close()
txtfile.close()
