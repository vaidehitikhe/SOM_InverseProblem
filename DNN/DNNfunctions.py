import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve,auc


# ------------------------------------------------------------
# Setup the color schemes and labels
# 
# ------------------------------------------------------------
print("hehe")
mybins = np.arange(0,1.05,0.1)

neuron_colors = ['xkcd:pinkish purple', 'xkcd:tangerine', 'xkcd:murky green', 'black']
process_colors = ['teal', 'blue', 'red', 'black']
validation_colors=['xkcd:royal blue','xkcd:red','xkcd:green','xkcd:grey']
df_names = ['M500', 'M1000', 'M1500','SM']

alpha_neuron_scores = [0.5, 0.5, 0.5, 0.3]

neuron_label_names = ['$n500$', '$n1000$', '$n1500$', '$nSM$']


# ------------------------------------------------------------
# Begin Function Definitions
# 
# ------------------------------------------------------------

def normalise_df(df, arg1, arg2):

    #Standard Normalisation
    #arg1 = meanvals, arg2 = stdvals

    normeddf = (df - arg1)/arg2

    return normeddf


def plot_process_scores(nnscore, dfname, output_file, include_bkg = True):

    plt.figure(figsize=(8,8))
    df_length = nnscore.shape[0]
    num_neurons = 3

    if include_bkg: num_neurons = 4

    for ind in range(0,num_neurons):
        
        plt.hist(nnscore[:,ind],bins=mybins,histtype='step',label=neuron_label_names[ind],linewidth=3,
                 color=neuron_colors[ind],density=False,log=False, weights=1/df_length*np.ones(df_length), alpha = alpha_neuron_scores[ind])
    
    plt.legend(loc='upper center')
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    plt.title(f'NN Output for {dfname}',fontsize=20)

    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim([0, 1])
    plt.savefig(output_file, format='pdf', pad_inches=0)
    plt.close()



def plot_neuron_scores(nnscore_list, dfnames, proclabel, output_file, include_bkg = True):
    ### proclabel: {0: 'M500', 1:'M1000', 2:'M1500', 3:'SM'}
    
    plt.figure(figsize=(8,8))
    num_neurons = 3
    if include_bkg: num_neurons = 4
    for ind in range(0,num_neurons):
        df_length = nnscore_list[ind].shape[0]
        ### Plotting the neuron score for different processes
        plt.hist(nnscore_list[ind][:,proclabel],bins=mybins,histtype='step',label=dfnames[ind],
                 linewidth=3,color=process_colors[ind],density=False,log=False, weights=1/df_length*np.ones(df_length))
        
    plt.legend(loc='upper center')
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    plt.title(f'{neuron_label_names[proclabel]} NN Output',fontsize=20)

    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim([0, 1])
    plt.savefig(output_file, format='pdf', bbox_inches = 'tight', pad_inches=0)    
    plt.close()

def write_fpr_tpr_text(fpr_vals, tpr_vals, label):
    roc_num_df = pd.DataFrame()
    roc_num_df['fpr'] = fpr_vals
    roc_num_df['tpr'] = tpr_vals
    roc_num_df.to_csv(label, sep=' ', header=False, index=False)

    
def plot_roc_curve(true_values, pred_values, dfnames, output_file, include_bkg = True):

    plt.figure(figsize=(8,8))
    num_neurons = 3
    if include_bkg: num_neurons = 4
    #text_op_names = ['FprTpr500_DNN.txt', 'FprTpr1000_DNN.txt', 'FprTpr1500_DNN.txt', 'FprTprSM_DNN.txt']
    for ind in range(num_neurons):
        fpr, tpr, _ = roc_curve(true_values[:, ind], pred_values[:, ind])
        #write_fpr_tpr_text(fpr, tpr, text_op_names[ind])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, color = neuron_colors[ind], label = neuron_label_names[ind]+ " (AUC = %0.3f)" % auc_score, lw=2)

    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha = 0.3) # Diagonal 'random guess' line

    plt.legend(loc='lower right', fontsize = 15)
    #plt.title(f'One vs Others-ROC Curve for Testing',fontsize=20)
    plt.xlabel('Background Efficiency',fontsize=20) #,fontsize=20
    plt.ylabel('Signal Efficiency',fontsize=20)
    #plt.xlim(0.,1.)
    #plt.ylim(0.,1.)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.savefig(output_file, format='pdf', bbox_inches = 'tight') #, pad_inches=0


def plot_training_roc_curve(train_true_values, train_pred_values, valid_true_values, valid_pred_values, dfnames, output_file, include_bkg = True):

    plt.figure(figsize=(8,8))
    num_neurons = 3
    if include_bkg: num_neurons = 4
    for ind in range(num_neurons):
        fpr_train, tpr_train, _ = roc_curve(train_true_values[:, ind], train_pred_values[:, ind])
        auc_score_train = auc(fpr_train, tpr_train)
        plt.plot(fpr_train, tpr_train, color = neuron_colors[ind],
                 label = neuron_label_names[ind]+ " Training (AUC = %0.3f)" % auc_score_train, lw = 2)

        fpr_valid, tpr_valid, _ = roc_curve(valid_true_values[:, ind], valid_pred_values[:, ind])
        auc_score_valid = auc(fpr_valid, tpr_valid)
        plt.plot(fpr_valid, tpr_valid, color = validation_colors[ind],
                 label = neuron_label_names[ind]+ " Validation (AUC = %0.3f)" % auc_score_valid)

    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha = 0.3) # Diagonal 'random guess' line
    plt.legend(loc='lower right',fontsize=15)
    #plt.title(f'One vs Others-ROC Curve for Training & Validation',fontsize=20)
    plt.xlabel('Background Efficiency',fontsize=17) 
    plt.ylabel('Signal Efficiency',fontsize=17)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    
    plt.savefig(output_file, format='pdf', bbox_inches = 'tight') #, pad_inches=0


def plot_one_process_score(nnscore, testdatalabel, output_file, with_bkg_score_cut=False):

    plt.figure(figsize=(8,8))
    df_length = nnscore.shape[0]
    median_score = np.median(nnscore, axis=0)
    mean_score = np.mean(nnscore, axis = 0)

    if with_bkg_score_cut:
        #Selecting events with bkg scores < 0.8
        nnscore = nnscore[nnscore[:,3]<0.8]
        df_length = nnscore.shape[0]
        median_score = np.median(nnscore, axis=0)
        mean_score = np.mean(nnscore, axis = 0)
        
    for ind in range(0,4):
        plt.hist(nnscore[:,ind],bins=mybins,histtype='step',label=neuron_label_names[ind] + " (%.3f)"%(median_score[ind]),linewidth=3,color=neuron_colors[ind],log=False, alpha = alpha_neuron_scores[ind])
        
    
    plt.legend(loc='upper center', fontsize = 15)
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    #plt.title(f'NN Output for {testdatalabel}',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks([0, 3, 6, 9, 12, 15, 18], fontsize=15)
    #plt.ylim([0, 1])
    plt.savefig(output_file, format='pdf', bbox_inches = 'tight')
    plt.close()
    
####### End Function definitons #######
