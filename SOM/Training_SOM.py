#A particle is something which can be treated as a single entity
#In this script, the model is trained and saved.
#100 x 100 and 40 x 40 TRAINING

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import pickle
from math import sqrt

import os
import warnings
warnings.filterwarnings('ignore')

from Training_Functions import *

print("Training functions - uses 9.5k training text file - for 40x40 SOM model")
print()

filenum = input("Enter a pdf and model number: ")
print()

print("Pdf will be saved as: VarySampSize__SOM_100train_<number entered>.pdf")
print("Model will be saved as: VarySampSize__SOM_100train_<number entered>.p")

pdf_name = "VarySampSize_SOM_100train_" + filenum + ".pdf"
pdf_path = os.path.join("train_pdf/" + pdf_name)
pp = PdfPages(pdf_path)

model_name = "VarySampSize_SOM_100train_" + filenum + ".p"
model = os.path.join("train_models/" + model_name)

#Only the plots of variables which will be used to train SOM
plotnames = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L']
mybins = np.arange(0,4000,40)
PlotVars(mybins, plotnames)
#plt.savefig(pp, format = 'pdf')

#label_names = {0: 'M500', 1: 'M1000', 2: 'M1500'}
label_names = {0: r"$m_{L} = 500$", 1: r"$m_{L} = 1000$", 2: r"$m_{L} = 1500$"}
color_list = {0: 'teal', 1: 'blue', 2: 'red'}
marker_list = {0: 'o', 1: '^', 2: 's'} #Different markers for different mass points

#AllTrainEv.to_csv("output2", sep=" ", index=True)

#Model training
from minisom import MiniSom
som_dim = (100, 100)
radius = 12
learn_rate = 1
epochs = 2000
som = MiniSom(som_dim[0], som_dim[1], Train_val.shape[1], sigma = radius, learning_rate = learn_rate, neighborhood_function = 'gaussian', random_seed = 0, topology = 'rectangular')
som.pca_weights_init(Train_val)
som.train(Train_val, epochs, verbose=True)

#BMUs
wx, wy = zip(*[som.winner(d) for d in Train_val])
wx = np.array(wx)
wy = np.array(wy)
winNeu = np.column_stack((wx, wy))

Train_df = np.column_stack((Train_val, wx, wy, Train_lab))
Train_df = pd.DataFrame(Train_df)
Train_df.columns = ['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'w_x', 'w_y', 'Label']

#Calculating SepScores
score_train = []
SepScore(score_train, Train_df, 'Training', som_dim[0], som_dim[1])
plt.savefig(pp, format = 'pdf')

plt.figure(figsize=(8.27, 11.69))
text1 = f"------------------ Model Summary -----------------" '\n' " " '\n' f"Dimensions of SOM model = {som_dim[0]} x {som_dim[1]}" '\n' f"Radius = {radius}" '\n' f"Learning rate = {learn_rate}" '\n' f"Epochs = {epochs}" '\n' f"Number of events of each M for training= 200" '\n' " " '\n'
plt.text(0.5, 0.90, text1, ha='center', va='top', wrap=True, fontsize=12)
plt.axis('off')
plt.savefig(pp, format = 'pdf')

#Plotting SOM sample distribution
plt.figure(figsize=(20,20))
#plt.title("Training Sample distribution", fontsize=12)
plt.pcolor(som.distance_map().T, cmap = 'bone_r', alpha = 0)
#plt.colorbar()

for c in np.unique(Train_lab):
    idx_label = Train_lab == c
    plt.scatter(wx[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, wy[idx_label]+0.5+(np.random.rand(np.sum(idx_label))-0.5)*0.8, s=100, c = color_list[c], label = label_names[c], marker = marker_list[c], alpha = 0.4)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_minor_locator(MultipleLocator(2))
#ax.yaxis.set_minor_locator(MultipleLocator(2))
plt.grid(True, which = 'major', linestyle = '-', linewidth = 1, alpha = 0.9)
plt.tick_params(axis='both', labelsize=25)
#plt.title('Training_SOM_v1', fontsize=15)
plt.legend(loc = 'lower right', fontsize=30)
plt.xlabel('Neuron X', fontsize=35)
plt.ylabel('Neuron Y', fontsize=35)
plt.savefig(pp, bbox_inches = 'tight', pad_inches=0, format = 'pdf')


#Saving SOM model
with open(model, 'wb') as outfile:
    pickle.dump(som, outfile)

print()
print('DONE!! Model is trained and pdf is saved')

pp.close()

os.system(f'xdg-open "train_pdf/{pdf_name}"')
