'''
This script uses 2k events and generates the baseline SOM test plot (M500, M1000, M1500, SM)
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import pickle
from math import sqrt
from minisom import MiniSom
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from Test_functions import *

print()
print('This takes 2k events of each M500, M1000, M1500, SM and tests the trained SOM model')


pdfnum = input("Enter a pdf to be saved: ")
pdf_name = "Final_test_100SOM_2k_" + pdfnum + ".pdf"
#pdf_name = "Final_test_40SOM_2k_" + pdfnum + ".pdf"
pdf_path = os.path.join("test_pdf/", pdf_name)
pp = PdfPages(pdf_path)

#Load a trained SOM:
modelnum = input('Enter the model (pre-trained) number: ')
model_name = "Final_SOM_100train_" + modelnum + ".p"
#model_name = "Final_SOM_40train_" + modelnum + ".p"
model_path = os.path.join("../trainSOM/train_models/", model_name)
with open(model_path, 'rb') as f:
    som = pickle.load(f)


som_dim = (100, 100)
#som_dim = (40, 40)

#Test_df
AllTestEv = pd.concat([M500_2kEv, M1000_2kEv, M1500_2kEv, SM_2kEv])
AllTestEv = AllTestEv.sample(frac=1).reset_index(drop=True)
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
    Test_df.to_pickle("pklFiles/Test_df_2kEvEach100_" + pdfnum + ".pkl")
    #Test_df.to_pickle("pklFiles/Test_df_2kEvEach40_" + pdfnum + ".pkl")
    print("Test_df of 2k Ev each saved as pickle!")

#In SOM_cases load this as Test_df = pd.read_pickle("Test_df_2kEvEach.pkl")

Test_sum = []
SepScore(Test_sum, Test_df, "Testing_2kEach", 100, 100)
#SepScore(Test_sum, Test_df, "Testing_2kEach", 40, 40)
plt.savefig(pp, format = 'pdf')

plt.figure(figsize=(20,20))
#plt.title('Baseline: 2k testing signal + 2k SM (1k + 1k)')
plt.pcolor(som.distance_map().T, cmap = 'bone_r', alpha=0.0)

color_list = {0: 'teal', 1: 'blue', 2: 'red', 3: 'purple', 4: 'black', 5: 'orange'}
#label_list = {0: 'M500', 1: 'M1000', 2: 'M1500', 3: 'M750', 4: 'SM', 5: 'M2500'}
label_list = {0: r"$m_{L} = 500$", 1: r"$m_{L} = 1000$", 2: r"$m_{L} = 1500$", 3: r"$m_{L} = 500$", 4: 'SM', 5: r"$m_{L} = 2500$"}
#marker_list = {0: 'o', 1: '^', 2: 's', 3: 'P', 4: 'x', 5: '*'}
marker_list = {0: 'o', 1: '^', 2: 's', 3: 'P', 4: 'X', 5: '*'}

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
#ax.xaxis.set_minor_locator(MultipleLocator(2))
#ax.yaxis.set_minor_locator(MultipleLocator(2))
plt.grid(True, which = 'major', linestyle = '-', linewidth = 1, alpha = 0.9)
plt.tick_params(axis='both', labelsize=25)
#plt.title('Testing 100x100 som with sm', fontsize=15)
plt.legend(loc = 'upper left', fontsize=30)
plt.xlabel('Neuron X', fontsize=35)
plt.ylabel('Neuron Y', fontsize=35)
plt.savefig(pp, bbox_inches = 'tight', pad_inches=0, format = 'pdf')

print()
print('SOM model tested using 2k events of each M500, M1000, M1500, SM!!')


pp.close()

os.system(f'xdg-open "test_pdf/{pdf_name}"')
