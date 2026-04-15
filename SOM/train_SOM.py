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

from SOMfunctions_train import *



# ------------------------------------------------------------
# Setup the names of the output file
# One PDF file, and the SOM model file
# ------------------------------------------------------------
print("Pdf will be saved as: outputs/SOM_train_<number entered>.pdf")
print("Model will be saved as: outputs/SOM_train_<number entered>.p")

filenum = input("Enter a number: ")
print()


pdf_name = "SOM_train_" + filenum + ".pdf"
pdf_path = os.path.join("outputs/", pdf_name)
pp = PdfPages(pdf_path)

model_name = "SOM_train_" + filenum + ".p"
model = os.path.join("outputs/", model_name)



# ------------------------------------------------------------
# Plot variables used to train the SOM
# 
# ------------------------------------------------------------
plotnames = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L']
mybins = np.arange(0,4000,40)
PlotVars(mybins, plotnames, pp)

# ------------------------------------------------------------
# Setup the colors
# 
# ------------------------------------------------------------
label_names = {0: 'M500', 1: 'M1000', 2: 'M1500'}
color_list = {0: 'teal', 1: 'blue', 2: 'red'}
marker_list = {0: 'o', 1: '^', 2: 's'} #Different markers for different mass points


# ------------------------------------------------------------
# Train the SOM
# 
# ------------------------------------------------------------
from minisom import MiniSom
som_dim = (100,100)
radius = 8
learn_rate = 1
epochs = 2000
som = MiniSom(som_dim[0], som_dim[1], Train_val.shape[1], sigma = radius, learning_rate = learn_rate, neighborhood_function = 'gaussian', random_seed = 0, topology = 'rectangular')
som.pca_weights_init(Train_val)
som.train(Train_val, epochs, verbose=True)
#Saving SOM model
with open(model, 'wb') as outfile:
    pickle.dump(som, outfile)

# ------------------------------------------------------------
# Finding the BMUs
# 
# ------------------------------------------------------------
wx, wy = zip(*[som.winner(d) for d in Train_val])
wx = np.array(wx)
wy = np.array(wy)
winNeu = np.column_stack((wx, wy))

Train_df = np.column_stack((Train_val, wx, wy, Train_lab))
Train_df = pd.DataFrame(Train_df)
Train_df.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'w_x', 'w_y', 'Label']

# ------------------------------------------------------------
# Calculating the separation scores and plotting the
#  events in the SOM.
# ------------------------------------------------------------
SepScore500,SepScore1000,SepScore1500 = SepScore(Train_df, 100, 100)
print(f'The separation scores are M500: {SepScore500:.3f}, M1000: {SepScore1000:.3f}, M1500: {SepScore1500:.3f}')

plt.figure(figsize=(20,20))
plt.pcolor(som.distance_map().T, cmap = 'bone_r', alpha = 0)
#plt.colorbar()

for c in np.unique(Train_lab):
    idx_label = Train_lab == c
    plt.scatter(wx[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, wy[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, s=50, c = color_list[c], label = label_names[c], marker = marker_list[c], alpha = 0.6)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))
plt.grid(True, which = 'major', linestyle = '-', linewidth = 1, alpha = 0.4)
plt.legend(loc = 'upper right', fontsize=30)
plt.xlabel('Neuron X', fontsize=30)
plt.ylabel('Neuron Y', fontsize=30)
plt.savefig(pp, format = 'pdf')
plt.close()


# ------------------------------------------------------------
pp.close()
