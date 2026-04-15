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
print("Pdf will be saved as: outputs/SOM_cases_<number entered>.pdf")
filenum = input("Enter a number: ")
print()
pdf_name = "SOM_cases_" + filenum + ".pdf"
pdf_path = os.path.join("outputs/", pdf_name)
pp = PdfPages(pdf_path)

modelnum = input("Enter the model (pre-trained number): ")
model_name = "SOM_train_" + modelnum + ".p"
model_path = os.path.join("outputs/", model_name)
with open(model_path, 'rb') as f:
    som = pickle.load(f)


som_dim = (100, 100)

Test_df = pd.read_pickle("Test_df_2kEvEach.pkl")

#---------------------------------------------------------------------------------------------------
#Case 1: Events of signal used to train the model (10 events of M1000)
#---------------------------------------------------------------------------------------------------
#case1_tot = M1000_500Ev.sample(n=10)
case1_tot = M1000_500Ev.head(n=10)
case1_tot = case1_tot[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]
case1_val = case1_tot[case1_tot.columns[:-1]].values
case1_lab = case1_tot['Label'].values
case1 = (case1_val - Train_mean) / Train_sd

#100x100
case1_sum = []
wx_c1, wy_c1 = zip(*[som.winner(d) for d in case1])
c1_reg5 = [[0,0,0,0] for _ in range(len(case1))]
scores_c1_reg5 = [[0,0,0,0] for _ in range(len(case1))]
c1_reg11 = [[0,0,0,0] for _ in range(len(case1))]
scores_c1_reg11 = [[0,0,0,0] for _ in range(len(case1))]

caseFunct(1, wx_c1,wy_c1,case1,case1_lab,pp,som.distance_map().T,2,Test_df)

print()
print('Case1: 10Ev M1000 DONE! Plots made!!')

#---------------------------------------------------------------------------------------------------
#Case 3:  A mixture two classes of signals, one of which is used to train the model (10 events of
#SM +10 events of M500)
#---------------------------------------------------------------------------------------------------
#case3_M500 = M500_500Ev.sample(n=10)
case3_M500 = M500_500Ev.head(n=10)
#case3_TTZ = TTZ_500Ev.sample(n=5)
#case3_WZ = WZ_500Ev.sample(n=5)
case3_TTZ = TTZ_500Ev.head(n=5)
case3_WZ = WZ_500Ev.head(n=5)
case3_tot = pd.concat([case3_M500, case3_TTZ, case3_WZ])

case3_sum = []
case3_tot = case3_tot[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]
case3_val = case3_tot[case3_tot.columns[:-1]].values
case3_lab = case3_tot['Label'].values
case3 = (case3_val - Train_mean) / Train_sd

wx_c3, wy_c3 = zip(*[som.winner(d) for d in case3])
c3_reg5 = [[0,0,0,0] for _ in range(len(case3))]
scores_c3_reg5 = [[0,0,0,0] for _ in range(len(case3))]
c3_reg11 = [[0,0,0,0] for _ in range(len(case3))]
scores_c3_reg11 = [[0,0,0,0] for _ in range(len(case3))]

caseFunct(3, wx_c3, wy_c3, case3, case3_lab, pp, som.distance_map().T, 2, Test_df)

print()
print('Case3: 10 SM + 10Ev M500 DONE! Plots made!!')




# ------------------------------------------------------------
pp.close()
