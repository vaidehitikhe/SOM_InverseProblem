'''
This is functions for the 4 cases. The 4 cases are as follows:
Case 1: Events of signal used to train the model (10 events of M1000)
Case 2: Events of signal not used to train the model (10 events of M2500)
Case 3: A mixture two classes of signals, one of which is used to train the model (10 events of
SM +10 events of M500)
Case 4: A mixture of two classes of signals, none of which is used to train the model (5 events
of SM + 10 events of M750)

**ADDED CASE:
Case 5: A mixture of two classes of signals, none of which is used to train the model (5 events of SM + 10 events of M600)
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
print('This considers 500 events of each M500, M750, M1000, M2500 and SM and uses these random sampling for 4 cases')


pdfnumb100 = input('Enter a pdf number to be generated for 100x100: ')
pdf_named100 = 'Cases_100SOM' + pdfnumb100 + '.pdf'
pdf_paths100 = os.path.join('outputs/', pdf_named100)
pp100 = PdfPages(pdf_paths100)

pdfnumb40 = input('Enter a pdf number to be generated for 40x40: ')
pdf_named40 = 'Cases_40SOM' + pdfnumb40 + '.pdf'
pdf_paths40 = os.path.join('outputs/', pdf_named40)
pp40 = PdfPages(pdf_paths40)


#Loading a pretrained model
#modelnum100 = input('Enter the model (pre-trained) number100x100: ')
#modelnum40 = input('Enter the model (pre-trained) number40x40: ')

#model_name100 = "Final_SOM_100train_" + modelnum100 + ".p"
#model_name40 = "Final_SOM_40train_" + modelnum40 + ".p"

model_name100 = "Train_100SOM.p"
model_name40 = "Train_40SOM.p"

model_path100 = os.path.join(model_name100)
with open(model_path100, 'rb') as f:
    som = pickle.load(f)
model_path40 = os.path.join(model_name40)
with open(model_path40, 'rb') as g:
    som40 = pickle.load(g)

som_dim = (100,100)
som_dim1 = (40,40)

#Test_df = pd.read_pickle(f"pklFiles/Test_df_2kEvEach100_{modelnum100}.pkl")
#Test_df40 = pd.read_pickle(f"pklFiles/Test_df_2kEvEach40_{modelnum40}.pkl")

Test_df = pd.read_pickle(f"outputs/Test_100SOM.pkl")
Test_df40 = pd.read_pickle(f"outputs/Test_40SOM.pkl")


plt.figure(figsize=(8.27,11.69))
text1 = f"Case 1: Events of signal used to train the model (10 events of M1000)" '\n' f"Case 2: Events of signal not used to train the model (10 events of M2500)" '\n' f"Case 3: A mixture two classes of signals, one of which is used to train the model (10 events of SM +10 events of M500)" '\n' f"Case 4: A mixture of two classes of signals, none of which is used to train the model (5 events of SM + 10 events of M750)" '\n' f"Case 5: A mixture of two classes of signals, none of which is used to train the model (5 events of SM + 10 events of M600)" '\n' " " '\n' f"I have sampled n number of events for each of the above case. For eg. M1000_df = M1000_500Ev.sample(n=10)" '\n' " " '\n' " " '\n' f"Note - This pdf contains plots for each case with the following order: Case number, SOM distribution, m1 x m1 score histogram without SM score cut, m1 x m1 Scores (w/o cut and with cut), m1 x m1 with SM score cut, trilep mass plot, m2 x m2 score histogram without SM score cut, m2 x m2 scores (w/o and with SM cut), m2 x m2 scores with SM score cut, trilep mass plot (same as above)." '\n' " " '\n' f"For 40 x 40 SOM model, m1 = 3 and m2 = 5" '\n' f"For 100 x 100 SOM model, m1 = 5, m2 = 11" '\n'
plt.text(0.5, 0.95, text1, ha = 'center', va = 'top', wrap = True, fontsize=12)
plt.axis('off')
plt.savefig(pp100, format = 'pdf')
plt.savefig(pp40, format = 'pdf')



#---------------------------------------------------------------------------------------------------
#Case 1: Events of signal used to train the model (10 events of M1000)
#---------------------------------------------------------------------------------------------------
case1_tot = M1000_500Ev.sample(n=10)
#case1_tot = M1000_500Ev.head(n=10)
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

caseFunct(1, wx_c1, wy_c1, case1_val, case1, case1_lab, case1_sum, 'C1_10M1000', pp100, 'C1_10M1000', som.distance_map().T, 2, 5, c1_reg5, c1_reg11, scores_c1_reg5, scores_c1_reg11, Test_df, 100, 100)

#40x40
case1_sum40 = []
wx_c1_40, wy_c1_40 = zip(*[som40.winner(d) for d in case1])
c1_reg3_40 = [[0,0,0,0] for _ in range(len(case1))]
scores_c1_reg3_40 = [[0,0,0,0] for _ in range(len(case1))]
c1_reg5_40 = [[0,0,0,0] for _ in range(len(case1))]
scores_c1_reg5_40 = [[0,0,0,0] for _ in range(len(case1))]

caseFunct(1, wx_c1_40, wy_c1_40, case1_val, case1, case1_lab, case1_sum40, 'C1_10M1000_40', pp40, 'C1_10M1000_40', som40.distance_map().T, 1, 2, c1_reg3_40, c1_reg5_40, scores_c1_reg3_40, scores_c1_reg5_40, Test_df40, 40, 40)


print()
print('Case1: 10Ev M1000 DONE! Plots made!!')

#---------------------------------------------------------------------------------------------------
#Case 2: Events of signal not used to train the model (10 events of M2500)
#---------------------------------------------------------------------------------------------------
case2_tot = M2500_500Ev.sample(n=10)
#case2_tot = M2500_500Ev.head(n=10)
case2_tot = case2_tot[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]
case2_val = case2_tot[case2_tot.columns[:-1]].values
case2_lab = case2_tot['Label'].values
case2 = (case2_val - Train_mean) / Train_sd

#100 x 100
case2_sum = []
wx_c2, wy_c2 = zip(*[som.winner(d) for d in case2])
c2_reg5 = [[0,0,0,0] for _ in range(len(case2))]
scores_c2_reg5 = [[0,0,0,0] for _ in range(len(case2))]
c2_reg11 = [[0,0,0,0] for _ in range(len(case2))]
scores_c2_reg11 = [[0,0,0,0] for _ in range(len(case2))]

caseFunct(2, wx_c2, wy_c2, case2_val, case2, case2_lab, case2_sum, 'C2_10M2500', pp100, 'C2_10M2500', som.distance_map().T, 2, 5, c2_reg5, c2_reg11, scores_c2_reg5, scores_c2_reg11, Test_df, 100, 100)

#40x40
case2_sum40 = []
wx_c2_40, wy_c2_40 = zip(*[som40.winner(d) for d in case2])
c2_reg3_40 = [[0,0,0,0] for _ in range(len(case2))]
scores_c2_reg3_40 = [[0,0,0,0] for _ in range(len(case2))]
c2_reg5_40 = [[0,0,0,0] for _ in range(len(case2))]
scores_c2_reg5_40 = [[0,0,0,0] for _ in range(len(case2))]

caseFunct(2, wx_c2_40, wy_c2_40, case2_val, case2, case2_lab, case2_sum40, 'C2_10M2500_40', pp40, 'C2_10M2500_40', som40.distance_map().T, 1, 2, c2_reg3_40, c2_reg5_40, scores_c2_reg3_40, scores_c2_reg5_40, Test_df40, 40, 40)


print()
print('Case2: 10Ev M2500 DONE! Plots made!!')

#---------------------------------------------------------------------------------------------------
#Case 3:  A mixture two classes of signals, one of which is used to train the model (10 events of
#SM +10 events of M500)
#---------------------------------------------------------------------------------------------------
case3_M500 = M500_500Ev.sample(n=10)
#case3_M500 = M500_500Ev.head(n=10)
case3_TTZ = TTZ_500Ev.sample(n=5)
case3_WZ = WZ_500Ev.sample(n=5)
#case3_TTZ = TTZ_500Ev.head(n=5)
#case3_WZ = WZ_500Ev.head(n=5)
case3_tot = pd.concat([case3_M500, case3_TTZ, case3_WZ])
case3_tot = case3_tot.sample(frac=1).reset_index(drop=True)


case3_sum = []
case3_tot = case3_tot[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]
case3_val = case3_tot[case3_tot.columns[:-1]].values
case3_lab = case3_tot['Label'].values
case3 = (case3_val - Train_mean) / Train_sd

#100 x 100
wx_c3, wy_c3 = zip(*[som.winner(d) for d in case3])
c3_reg5 = [[0,0,0,0] for _ in range(len(case3))]
scores_c3_reg5 = [[0,0,0,0] for _ in range(len(case3))]
c3_reg11 = [[0,0,0,0] for _ in range(len(case3))]
scores_c3_reg11 = [[0,0,0,0] for _ in range(len(case3))]

caseFunct(3, wx_c3, wy_c3, case3_val, case3, case3_lab, case3_sum, 'C3_10SM_10M500', pp100, 'C3_10SM_10M500', som.distance_map().T, 2, 5, c3_reg5, c3_reg11, scores_c3_reg5, scores_c3_reg11, Test_df, 100, 100)

#40x40
case3_sum40 = []
wx_c3_40, wy_c3_40 = zip(*[som40.winner(d) for d in case3])
c3_reg3_40 = [[0,0,0,0] for _ in range(len(case3))]
scores_c3_reg3_40 = [[0,0,0,0] for _ in range(len(case3))]
c3_reg5_40 = [[0,0,0,0] for _ in range(len(case3))]
scores_c3_reg5_40 = [[0,0,0,0] for _ in range(len(case3))]

caseFunct(3, wx_c3_40, wy_c3_40, case3_val, case3, case3_lab, case3_sum40, 'C3_10SM_10M500_40', pp40, 'C3_10SM_10M500_40', som40.distance_map().T, 1, 2, c3_reg3_40, c3_reg5_40, scores_c3_reg3_40, scores_c3_reg5_40, Test_df40, 40, 40)


print()
print('Case3: 10 SM + 10Ev M500 DONE! Plots made!!')


#---------------------------------------------------------------------------------------------------
#Case 4:  A mixture of two classes of signals, none of which is used to train the model (5 events
#of SM + 10 events of M750)
#---------------------------------------------------------------------------------------------------
case4_M750 = M750_500Ev.sample(n=10)
#case4_M750 = M750_500Ev.head(n=10)
case4_TTZ = TTZ_500Ev.sample(n=3)
case4_WZ = WZ_500Ev.sample(n=2)
#case4_TTZ = TTZ_500Ev.iloc[15:18] #15th row excluded, 18th row included
#case4_WZ = WZ_500Ev.iloc[15:17]
case4_tot = pd.concat([case4_M750, case4_TTZ, case4_WZ])
case4_tot = case4_tot.sample(frac=1).reset_index(drop=True)


case4_sum = []
case4_tot = case4_tot[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]
case4_val = case4_tot[case4_tot.columns[:-1]].values
case4_lab = case4_tot['Label'].values
case4 = (case4_val - Train_mean) / Train_sd

#100 x 100
wx_c4, wy_c4 = zip(*[som.winner(d) for d in case4])
c4_reg5 = [[0,0,0,0] for _ in range(len(case4))]
scores_c4_reg5 = [[0,0,0,0] for _ in range(len(case4))]
c4_reg11 = [[0,0,0,0] for _ in range(len(case4))]
scores_c4_reg11 = [[0,0,0,0] for _ in range(len(case4))]

caseFunct(4, wx_c4, wy_c4, case4_val, case4, case4_lab, case4_sum, 'C4_5SM_10M750', pp100, 'C4_5SM_10M750', som.distance_map().T, 2, 5, c4_reg5, c4_reg11, scores_c4_reg5, scores_c4_reg11, Test_df, 100, 100)

#40x40
case4_sum40 = []
wx_c4_40, wy_c4_40 = zip(*[som40.winner(d) for d in case4])
c4_reg3_40 = [[0,0,0,0] for _ in range(len(case4))]
scores_c4_reg3_40 = [[0,0,0,0] for _ in range(len(case4))]
c4_reg5_40 = [[0,0,0,0] for _ in range(len(case4))]
scores_c4_reg5_40 = [[0,0,0,0] for _ in range(len(case4))]

caseFunct(4, wx_c4_40, wy_c4_40, case4_val, case4, case4_lab, case4_sum40, 'C4_5SM_10M750_40', pp40, 'C4_5SM_10M750_40', som40.distance_map().T, 1, 2, c4_reg3_40, c4_reg5_40, scores_c4_reg3_40, scores_c4_reg5_40, Test_df40, 40, 40)


print()
print('Case4: 5 SM + 10Ev M750 DONE! Plots made!!')

#------------------------------------------------------------------------------------------------------------------
#ADDED CASES

#------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------
#Case 6:  A mixture of two classes of signals, none of which is used to train the model (5 events
#of SM + 10 events of M600)
#---------------------------------------------------------------------------------------------------
case5_M600 = M600_500Ev.sample(n=10)

#case6_M600 = M600_500Ev.head(n=10)

case5_TTZ = TTZ_500Ev.sample(n=3)
case5_WZ = WZ_500Ev.sample(n=2)

#case5_TTZ = TTZ_500Ev.iloc[15:18] #15th row excluded, 18th row included
#case5_WZ = WZ_500Ev.iloc[15:17]

case5_tot = pd.concat([case5_M600, case5_TTZ, case5_WZ])
case5_tot = case5_tot.sample(frac=1).reset_index(drop=True)


case5_sum = []
case5_tot = case5_tot[['LT_3L', 'HT_3L', 'Mos1_3L', 'trilepM', 'highmT_3L', 'pT0j0pT', 'Mos2_3L', '3lmT_3L', 'Label']]
case5_val = case5_tot[case5_tot.columns[:-1]].values
case5_lab = case5_tot['Label'].values
case5 = (case5_val - Train_mean) / Train_sd

#100 x 100
wx_c5, wy_c5 = zip(*[som.winner(d) for d in case5])
c5_reg5 = [[0,0,0,0] for _ in range(len(case5))]
scores_c5_reg5 = [[0,0,0,0] for _ in range(len(case5))]
c5_reg11 = [[0,0,0,0] for _ in range(len(case5))]
scores_c5_reg11 = [[0,0,0,0] for _ in range(len(case5))]

caseFunct(5, wx_c5, wy_c5, case5_val, case5, case5_lab, case5_sum, 'C5_5SM_10M600', pp100, 'C5_5SM_10M600', som.distance_map().T, 2, 5, c5_reg5, c5_reg11, scores_c5_reg5, scores_c5_reg11, Test_df, 100, 100)

#40x40
case5_sum40 = []
wx_c5_40, wy_c5_40 = zip(*[som40.winner(d) for d in case5])
c5_reg3_40 = [[0,0,0,0] for _ in range(len(case5))]
scores_c5_reg3_40 = [[0,0,0,0] for _ in range(len(case5))]
c5_reg5_40 = [[0,0,0,0] for _ in range(len(case5))]
scores_c5_reg5_40 = [[0,0,0,0] for _ in range(len(case5))]

caseFunct(5, wx_c5_40, wy_c5_40, case5_val, case5, case5_lab, case5_sum40, 'C5_5SM_10M600_40', pp40, 'C5_5SM_10M600_40', som40.distance_map().T, 1, 2, c5_reg3_40, c5_reg5_40, scores_c5_reg3_40, scores_c5_reg5_40, Test_df40, 40, 40)

print()
print('Case5: 5 SM + 10Ev M600 DONE! Plots made!!')


#---------------------------------------------------------------------------------------------------------





pp100.close()
pp40.close()
