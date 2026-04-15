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
outputname= foldername+"train_test_plots.pdf"
textfilename= foldername+"summary_metrics.txt"


pp = PdfPages(outputname)

# ------------------------------------------------------------
# Define the names of the input variables
# from the two sets of files.
# ------------------------------------------------------------

col_names_bkg=['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'Mos1_3L', 'Mos2_3L', 
           'trilepM', 'mt0', 'mt1', 'mt2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 
           'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 
           'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l', '3lmT_3L', 'highmT_3L', 'pT0j0pT', 
           'j0pTbyHT', 'pT0byLT']

#This is the full list of input variables in the text file for the signal files
col_names = ['LT_3L', 'HT_3L', 'njets', 'j0pT', 'MET_3L', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'Mos1_3L', 'Mos2_3L', 'trilepM', 'mT0', 'mT1', 'mT2', 'highmT_3L', '3lmT_3L', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'pT0_3L', 'pT1_3L', 'pT2_3L' , 'pT_tau_3L']


#These are the variables we will be training on
train_vars_bkg = ['LT_3L', 'HT_3L', 'Mos1_3L', 'Mos2_3L', 'trilepM', '3lmT_3L', 'highmT_3L', 'pT0j0pT'] #Training variables
cols_bkg = [col_names_bkg.index(var) for var in train_vars_bkg]   #Getting the indices of the training varaibles

train_vars = ['LT_3L', 'HT_3L', 'Mos1_3L', 'Mos2_3L', 'trilepM', '3lmT_3L', 'highmT_3L','pT0_3L', 'j0pT'] #Not the Training variables
cols = [col_names.index(var) for var in train_vars]   #Getting the indices of the training varaibles

# ------------------------------------------------------------
# Load the dataframes
# 
# ------------------------------------------------------------

text_path = '../inputs'

VLL_m500_train = pd.read_csv(text_path + '/M500/M500_9500Ev_train.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m500_test = pd.read_csv(text_path + '/M500/M500_2000Ev_test.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m500_train['label']=0
VLL_m500_test['label']=0

VLL_m1000_train = pd.read_csv(text_path + '/M1000/M1000_9500Ev_train.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000_test = pd.read_csv(text_path + '/M1000/M1000_2000Ev_test.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000_train['label']=1
VLL_m1000_test['label']=1

VLL_m1500_train = pd.read_csv(text_path + '/M1500/M1500_9500Ev_train.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1500_test = pd.read_csv(text_path + '/M1500/M1500_2000Ev_test.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1500_train['label']=2
VLL_m1500_test['label']=2



all_sig_dfs = [VLL_m500_train, VLL_m1000_train, VLL_m1500_train, VLL_m500_test, VLL_m1000_test, VLL_m1500_test]

for df in all_sig_dfs:
    df['pT0j0pT'] = df['pT0_3L'] + df['j0pT']

train_vars = train_vars_bkg


WZ_bkg_train = pd.read_csv(text_path + '/WZ_train.txt',sep=' ',index_col=None, usecols=cols_bkg,names=col_names_bkg)
WZ_bkg_test = pd.read_csv(text_path + '/WZ_test.txt',sep=' ',index_col=None, usecols=cols_bkg,names=col_names_bkg)
WZ_bkg_train['label']=3
WZ_bkg_test['label']=3

TTZ_bkg_train = pd.read_csv(text_path + '/TTZ_train.txt',sep=' ',index_col=None, usecols=cols_bkg,names=col_names_bkg)
TTZ_bkg_test = pd.read_csv(text_path + '/TTZ_test.txt',sep=' ',index_col=None, usecols=cols_bkg,names=col_names_bkg)

TTZ_bkg_train["label"]=3
TTZ_bkg_test["label"]=3



print(VLL_m500_train.shape, VLL_m1000_train.shape, VLL_m1500_train.shape, WZ_bkg_train.shape, TTZ_bkg_train.shape)
print(VLL_m500_test.shape, VLL_m1000_test.shape, VLL_m1500_test.shape, WZ_bkg_test.shape, TTZ_bkg_test.shape)


# ------------------------------------------------------------
# Gathering the training and testing datasets
# and normalizing them
# ------------------------------------------------------------

train_data = pd.concat([VLL_m500_train, VLL_m1000_train, VLL_m1500_train, WZ_bkg_train, TTZ_bkg_train], ignore_index = True)
test_data = pd.concat([VLL_m500_test, VLL_m1000_test, VLL_m1500_test, WZ_bkg_test, TTZ_bkg_test], ignore_index = True)
X_train = train_data[train_vars]
X_test = test_data[train_vars]
y_train = train_data['label']
y_test = test_data['label']


### Changing class label from 0,1,2,3 to one hot encoded values (1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

### Calculating the min, max, mean, std values for the training dataset
maxValues = X_train.max(axis=0)
minValues = X_train.min(axis=0)
meanValues = X_train.mean(axis=0)
stdValues = X_train.std(axis=0)
print("Max values")
print(list(maxValues))
print("Min values")
print(list(minValues))
print("Mean values")
print(list(meanValues))
print("STD values")
print(list(stdValues))

#Mean-std scaling
arg1 = meanValues; arg2 = stdValues;

#Normalising the dataframe
X_train = normalise_df(X_train, arg1, arg2)
X_test = normalise_df(X_test, arg1, arg2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
n_features = X_train.shape[1]
print(f'The number of input variables is {n_features}')

# ------------------------------------------------------------
# Setting up the model and training it
# 
# ------------------------------------------------------------

model = Sequential()
model.add(Dense(32, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#### Training begins
start_time = time.time()
history = model.fit(X_train,y_train,epochs=200,batch_size=50,validation_data=(X_test,y_test),verbose=1, shuffle = True)
end_time = time.time()
#### Training is dome
print(f"\nTraining took {end_time - start_time} seconds!\n")

#Now we print the model summary to screen and save the model file
print('The NN architecture is')
model.summary()
model.save(modelname)


# ------------------------------------------------------------
# Now we do the prediction and plot the training ROC
# curve, and the other plots.
# ------------------------------------------------------------

train_pred_prob = model.predict(X_train)
valid_pred_prob = model.predict(X_test)

plot_training_roc_curve(y_train, train_pred_prob, y_test, valid_pred_prob, df_names, pp)

#### Getting the predicted scores for the individual processes
vll_m500_pred_scores = valid_pred_prob[y_test[:,0]==1]
vll_m1000_pred_scores = valid_pred_prob[y_test[:,1]==1]
vll_m1500_pred_scores = valid_pred_prob[y_test[:,2]==1]
bkg_pred_scores = valid_pred_prob[y_test[:,3]==1]

#### Ordered list of neuron scores for the different processes
NNscores_list = [vll_m500_pred_scores, vll_m1000_pred_scores, vll_m1500_pred_scores, bkg_pred_scores]

#### Plottig the process scores
plot_process_scores(vll_m500_pred_scores, 'procM500', pp)
plot_process_scores(vll_m1000_pred_scores, 'procM1000', pp)
plot_process_scores(vll_m1500_pred_scores, 'procM1500', pp)
plot_process_scores(bkg_pred_scores, 'procBkg', pp)

#### Plottig the neuron scores
plot_neuron_scores(NNscores_list, df_names, 0, pp)
plot_neuron_scores(NNscores_list, df_names, 1, pp)
plot_neuron_scores(NNscores_list, df_names, 2, pp)
plot_neuron_scores(NNscores_list, df_names, 3, pp)



#### Plotting the One vs Others ROC Curve
plot_roc_curve(y_test, valid_pred_prob, df_names, pp, include_bkg = False)

# ------------------------------------------------------------
# Print some information
# 
# ------------------------------------------------------------

with open(textfilename, "w") as txtfile:
    
    txtfile.write(f"Training Variables used: {train_vars}\n")
    txtfile.write(f"Maximum Values: {list(maxValues)}\n")
    txtfile.write(f"Minimum Values: {list(minValues)}\n")
    txtfile.write(f"Mean Values: {list(meanValues)}\n")
    txtfile.write(f"STD Values: {list(stdValues)}\n")

    
print(f'All done. Output is saved as {outputname} and {textfilename}')


# ------------------------------------------------------------
pp.close()
