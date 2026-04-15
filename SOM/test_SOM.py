#!/usr/bin/env python
# coding: utf-8

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

from SOMfunctions_test import *


# ------------------------------------------------------------
# Setup the names of the output PDF file
# and read the SOM model file
# ------------------------------------------------------------
print("Pdf will be saved as: outputs/SOM_test2k_<number entered>.pdf")
filenum = input("Enter a number: ")
print()
pdf_name = "SOM_test2k_" + filenum + ".pdf"
pdf_path = os.path.join("outputs/", pdf_name)
pp = PdfPages(pdf_path)

modelnum = input("Enter the model (pre-trained number): ")
model_name = "SOM_train_" + modelnum + ".p"
model_path = os.path.join("outputs/", model_name)
with open(model_path, 'rb') as f:
    som = pickle.load(f)


som_dim = (100, 100)


# ------------------------------------------------------------
# Find the BMUs for the 2000 event test dataset 
# 
# ------------------------------------------------------------

AllTestEv = pd.concat([M500_2kEv, M1000_2kEv, M1500_2kEv, SM_2kEv])
AllTestEv = AllTestEv[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]
#Separating the values and label column
Test_val_UN = AllTestEv[AllTestEv.columns[:-1]].values #UN stands for unnormalized
Test_val = (Test_val_UN - Train_mean) / Train_sd#Normalized
Test_lab = AllTestEv['Label'].values

wx_test, wy_test = zip(*[som.winner(d) for d in Test_val])
wx_test = np.array(wx_test)
wy_test = np.array(wy_test)

Test_df = np.column_stack((Test_val, wx_test, wy_test, Test_lab))
Test_df = pd.DataFrame(Test_df)
Test_df.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'w_x', 'w_y', 'Label']

#Saving the 'Test_df' as a pickle to load it into SOM_cases for cluster score eval.
if __name__ == "__main__":
    Test_df.to_pickle("Test_df_2kEvEach.pkl")
    #Test_df.to_pickle("Test_df_2kEvEach40.pkl")
    print("Test_df of 2k Ev each saved as pickle!")

#In SOM_cases load this as Test_df = pd.read_pickle("Test_df_2kEvEach.pkl")

# ------------------------------------------------------------
# Calculating the separation scores and plotting the
#  events in the SOM.
# ------------------------------------------------------------

SepScore500,SepScore1000,SepScore1500,SepScoreSM = SepScore(Test_df, 100, 100)
print(f'The separation scores are M500: {SepScore500:.3f}, M1000: {SepScore1000:.3f}, M1500: {SepScore1500:.3f}, SM: {SepScoreSM:.3f}')


color_list = {0: 'teal', 1: 'blue', 2: 'red', 3: 'purple', 4: 'black', 5: 'orange'}
label_list = {0: r"$m_{L} = 500$", 1: r"$m_{L} = 1000$", 2: r"$m_{L} = 1500$", 3: r"$m_{L} = 500$", 4: 'SM', 5: r"$m_{L} = 2500$"}
marker_list = {0: 'o', 1: '^', 2: 's', 3: 'P', 4: 'X', 5: '*'}


plt.figure(figsize=(20,20))
plt.pcolor(som.distance_map().T, cmap = 'bone_r', alpha=0.0)
idx_label1 = Test_lab == 4
plt.scatter(wx_test[idx_label1]+0.5+(np.random.rand(np.sum(idx_label1))-0.5)*0.8, wy_test[idx_label1]+0.5+(np.random.rand(np.sum(idx_label1))-0.5)*0.8, s=150, c=color_list[4], label = label_list[4], marker = marker_list[4], alpha = 0.8)
for e in np.unique(Test_lab):
    if e!=4:
        idx_label = Test_lab == e
        alp = 0.4
        plt.scatter(wx_test[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, wy_test[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, s=150, c=color_list[e], label = label_list[e], marker = marker_list[e], alpha = alp)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))
plt.grid(True, which = 'major', linestyle = '-', linewidth = 1, alpha = 0.9)
plt.tick_params(axis='both', labelsize=25)
plt.legend(loc = 'upper left', fontsize=30)
plt.xlabel('Neuron X', fontsize=35)
plt.ylabel('Neuron Y', fontsize=35)
plt.savefig(pp, format = 'pdf')

print()
print('SOM model tested using 2k events of each M500, M1000, M1500, SM!!')


# ------------------------------------------------------------
# Calculating the ROC curve
#  
# ------------------------------------------------------------
forRoc_tot = pd.concat([M500_500Ev, M1000_500Ev, M1500_500Ev, TTZ_500Ev, WZ_500Ev]) 
forRoc_tot = forRoc_tot[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]
forRoc_val_UN = forRoc_tot[forRoc_tot.columns[:-1]].values #Unnormalized
forRoc_lab = forRoc_tot['Label'].values

forRoc_val = (forRoc_val_UN - Train_mean) / Train_sd

wx_forRoc, wy_forRoc = zip(*[som.winner(d) for d in forRoc_val])

forRoc_reg = [[0,0,0,0] for _ in range(len(forRoc_val))]
forRoc_scores = [[0,0,0,0] for _ in range(len(forRoc_val))]

myres =caseFunct_mod(wx_forRoc, wy_forRoc, forRoc_val, forRoc_lab, 4, Test_df)
print()
print('forRoc case DONE!!')

plotROC(myres, pp)
print()
print('ROC plotted!!')





# ------------------------------------------------------------
pp.close()
