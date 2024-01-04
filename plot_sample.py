"""
a sample file for generating plots from the saved results. 
"""
#%%
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter


if not os.path.exists('./plots'):
    os.makedirs('./plots')

xticks_size = 14
yticks_size = 14
xlabel_size = 14
ylabel_size = 18
legend_size = 12
legend_loc = 'lower right'
linewidth = 3
linestyle = '--'
markersize = 10

def load_res(path):
    f = open(path,'rb')
    res = pkl.load(f)
    f.close 
    return res

def read_metric(skeleton_n_ci,metric='skel_precision'):
    skel_metric = [skeleton_n_ci[i][metric] for i in range(len(skeleton_n_ci))]
    return skel_metric

#%% Generating Fig.8(a) of the paper
# n = 8, varying mixing rate

mix_rate_list = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
n_samples_list = [1_000, 3_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000]
n_mix_vary = len(mix_rate_list)          # 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99
n_n_samples = len(n_samples_list)          # 1k, 3k, 5k, 10k, 20k, 30k, 40k, 50k

skeleton_8_mix_1k = load_res('./results/skeleton_8_mix_1k.pkl')
skeleton_8_mix_3k = load_res('./results/skeleton_8_mix_3k.pkl')
skeleton_8_mix_5k = load_res('./results/skeleton_8_mix_5k.pkl')
skeleton_8_mix_10k = load_res('./results/skeleton_8_mix_10k.pkl')
skeleton_8_mix_20k = load_res('./results/skeleton_8_mix_20k.pkl')
skeleton_8_mix_30k = load_res('./results/skeleton_8_mix_30k.pkl')
skeleton_8_mix_40k = load_res('./results/skeleton_8_mix_40k.pkl')
skeleton_8_mix_50k = load_res('./results/skeleton_8_mix_50k.pkl')

skeleton_8_mix = [skeleton_8_mix_1k,skeleton_8_mix_3k,skeleton_8_mix_5k,\
                  skeleton_8_mix_10k,skeleton_8_mix_20k,skeleton_8_mix_30k,\
                    skeleton_8_mix_40k,skeleton_8_mix_50k]

mix_keys = ['res_8_mix_50','res_8_mix_60','res_8_mix_70','res_8_mix_80',\
            'res_8_mix_90','res_8_mix_95','res_8_mix_99']

skel_precision_8 = np.zeros((n_mix_vary,n_n_samples))
skel_recall_8 = np.zeros((n_mix_vary,n_n_samples))
skel_u_recall_8 = np.zeros((n_mix_vary,n_n_samples))
skel_e_recall_8 = np.zeros((n_mix_vary,n_n_samples))

for idx_mix_rate in range(n_mix_vary):
    for idx_n_samples in range(n_n_samples):
        skel_precision_8[idx_mix_rate,idx_n_samples] = skeleton_8_mix[idx_n_samples][mix_keys[idx_mix_rate]]['skel_precision']      
        skel_recall_8[idx_mix_rate,idx_n_samples] = skeleton_8_mix[idx_n_samples][mix_keys[idx_mix_rate]]['skel_recall']      
        skel_u_recall_8[idx_mix_rate,idx_n_samples] = skeleton_8_mix[idx_n_samples][mix_keys[idx_mix_rate]]['skel_u_recall']      
        skel_e_recall_8[idx_mix_rate,idx_n_samples] = skeleton_8_mix[idx_n_samples][mix_keys[idx_mix_rate]]['skel_e_recall']      

#%%
# indices for n_samples to plot
idx_plot_n_samples = [1,3,5,7]
n_samples_8 = [3000, 10000, 30000, 50000]


plt.figure()
plt.title('$n=8$',fontsize = 20)
plt.plot(mix_rate_list,skel_precision_8[:,1],'-o',markersize=markersize,label='3k',linewidth=linewidth,linestyle=linestyle)
plt.plot(mix_rate_list,skel_precision_8[:,3],'-o',markersize=markersize,label='10k',linewidth=linewidth,linestyle=linestyle)
plt.plot(mix_rate_list,skel_precision_8[:,5],'-o',markersize=markersize,label='30k',linewidth=linewidth,linestyle=linestyle)
plt.plot(mix_rate_list,skel_precision_8[:,7],'-o',markersize=markersize,label='50k',linewidth=linewidth,linestyle=linestyle)

plt.ylim([0.9,1])
#plt.xscale('log')
plt.xlabel('Mixing rate',size=xlabel_size)
plt.xticks(mix_rate_list,fontsize=12)
plt.yticks(fontsize=yticks_size)
plt.legend(fontsize=legend_size,loc='lower left')
plt.tight_layout()
plt.grid()
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
plt.savefig('./plots/skeleton_precision_8_mixing_rate.pdf')
