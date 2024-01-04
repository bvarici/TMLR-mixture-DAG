"""
Use CI tests to learn the skeleton.
Use partial correlation as CI test.

Uncomment the corresponding parts for generating sample results for 
- main setting: partial correlation as CI test, 50/50 mixing rate between two DAGs
- alternative settings: changing the CI test, varying the mixing rate, changing the noise variance etc.
- for more than two component DAGs, refer to arbitrary_components.py file

"""
#%%
import warnings
warnings.simplefilter('ignore')
import numpy as np
import os
import causaldag as cd
from utils import generate_samples, is_forest, create_true_skeleton_mixture_graph, find_mixture_skeleton, GCM
import pickle as pkl


#%%
if not os.path.exists('./results'):
    os.makedirs('./results')

def run_once(N=6,avg_neighbor=2,n_samples=5000,mix_rate=0.5,ci_alpha=0.1,ci_test='partial_correlation',unequal_noise=False):
    # create random Erdos-Renyi DAGs, they should have forest structure
    forest_flag_1 = False
    forest_flag_2 = False
    while forest_flag_1 is False:
        # let G1 be have 1---n topological order WLOG
        A1 = np.triu(np.random.uniform(size=[N,N])<(avg_neighbor/N),k=1)
        if is_forest(A1) is True:
            forest_flag_1 = True
        # else:
        #     print('DAG1 is not forest, sampling another DAG')
    
    while forest_flag_2 is False:    
        # for A2, top order needs to be random.
        A2 = np.triu(np.random.uniform(size=[N,N])<(avg_neighbor/N),k=1)
        idx = np.random.permutation(N)
        A2 = A2[idx][:,idx]
        if is_forest(A2) is True:
            forest_flag_2 = True
        # else:
        #     print('DAG2 is not forest, sampling another DAG')
    
    G1 = cd.DAG.from_amat(A1)
    to1 = G1.topological_sort()
    G2 = cd.DAG.from_amat(A2)
    to2 = G2.topological_sort()
    
    # assign weights to the edges
    rand_weights = np.random.uniform(-2,-0.25,[N,N])* np.random.choice([-1,1],size=[N,N])
    W1 = A1*rand_weights
    W2 = A2*rand_weights
    delta = np.where(np.sum(A1!=A2,0))[0]
    non_delta = [i for i in range(N) if i not in delta]
    
    mu_epsilon = np.random.uniform(-2,2,size=N)
    if unequal_noise is True:
        sigma_epsilon_1 = np.random.uniform(size=N,low=0.5,high=1.5)
        sigma_epsilon_2 = np.random.uniform(size=N,low=0.5,high=1.5)
    else:
        sigma_epsilon_1 = np.ones(N)
        sigma_epsilon_2 = np.ones(N)
    
    #% generate samples
    n_samples_1 = int(n_samples * mix_rate)
    n_samples_2 = int(n_samples * (1 - mix_rate))
    data1 = generate_samples(W1,mu_epsilon,sigma_epsilon_1,n_samples_1,topological_order=to1)
    data2 = generate_samples(W2,mu_epsilon,sigma_epsilon_2,n_samples_2,topological_order=to2)
    datamix = np.vstack([data1,data2])
    np.random.shuffle(datamix)
    #suffstat = partial_correlation_suffstat(datamix)
    
    GM_skel, EM, sepsets = find_mixture_skeleton(datamix,ci_alpha,ci_test=ci_test)

    # call ground truth
    res = create_true_skeleton_mixture_graph(G1,G2) 
    GM_skel_GT = res['GM_skel']
    sepsets_GT = res['sepsets']
    EM_GT = res['EM']
    EU_GT = res['EU']
    EE_GT = res['EE']
    E1_GT = [(edge[0],edge[1]) if edge[0] < edge[1] else (edge[1],edge[0]) for edge in res['E1']]
    E2_GT = [(edge[0],edge[1]) if edge[0] < edge[1] else (edge[1],edge[0]) for edge in res['E2']]


    # overall
    skel_tp = len(set(EM_GT) & set(EM))
    skel_fp = len(set(EM)-set(EM_GT))
    skel_fn = len(set(EM_GT)-set(EM))
    # for union edges
    skel_u_tp = len(set(EU_GT)&set(EM))
    skel_u_fn = len(set(EU_GT)-set(EM))
    skel_e1_tp = len(set(E1_GT)&set(EM))
    skel_e1_fn = len(set(E1_GT)-set(EM))
    skel_e2_tp = len(set(E2_GT)&set(EM))
    skel_e2_fn = len(set(E2_GT)-set(EM))

    # for emergent edges
    skel_e_tp = len(set(EE_GT) & set(EM))
    skel_e_fn = len(set(EE_GT)-set(EM))

    
    return res, skel_tp, skel_fp, skel_fn, skel_u_tp, skel_u_fn, skel_e_tp, skel_e_fn


def run_repeat(n_repeat=10,N=6,avg_neighbor=2,n_samples=5000,mix_rate=0.5,ci_alpha=0.1,ci_test='partial_correlation',unequal_noise=False):
    n_skel_tp = 0
    n_skel_fp = 0
    n_skel_fn = 0
    n_skel_u_tp = 0
    n_skel_u_fn = 0
    n_skel_e_tp = 0
    n_skel_e_fn = 0    
    n_delta = 0

    for run_idx in range(n_repeat):
        res, skel_tp, skel_fp, skel_fn, skel_u_tp, skel_u_fn, skel_e_tp, skel_e_fn = \
                    run_once(N=N,avg_neighbor=avg_neighbor,n_samples=n_samples,mix_rate=mix_rate,ci_alpha=ci_alpha,ci_test=ci_test,unequal_noise=unequal_noise)
            
    
        n_delta += len(res['delta'])
        n_skel_tp += skel_tp
        n_skel_fp += skel_fp
        n_skel_fn += skel_fn
        n_skel_u_tp += skel_u_tp
        n_skel_u_fn += skel_u_fn
        n_skel_e_tp += skel_e_tp
        n_skel_e_fn += skel_e_fn

        print('run_idx:',run_idx)
    
    skel_precision = n_skel_tp / (n_skel_tp+n_skel_fp)
    skel_recall = n_skel_tp / (n_skel_tp+n_skel_fn)
    skel_f1 = 2 * skel_precision * skel_recall / (skel_precision + skel_recall)
    skel_u_recall = n_skel_u_tp / (n_skel_u_tp+n_skel_u_fn)
    skel_e_recall = n_skel_e_tp / (n_skel_e_tp+n_skel_e_fn)
    
    
    # separability results can be reported too, just the complementary ones
    n_pairs = n_repeat * N * (N-1) / 2
    n_inseparable_pairs = n_skel_tp + n_skel_fn
    n_separable_pairs = n_pairs - n_inseparable_pairs
    n_skel_tn = n_separable_pairs - n_skel_fp

    separable_precision = n_skel_tn / (n_skel_tn + n_skel_fn)
    separable_recall = n_skel_tn / (n_skel_tn + n_skel_fp)
    separable_f1 = 2*separable_precision*separable_recall / (separable_precision+separable_recall)
    
    res_repeat = {'skel_precision':skel_precision, 'skel_recall':skel_recall, 'skel_f1':skel_f1,\
                  'skel_u_recall':skel_u_recall,'skel_e_recall':skel_e_recall, \
                      'separable_precision':separable_precision, 'separable_recall':separable_recall,\
                          'separable_f1':separable_f1,'avg_delta':n_delta/n_repeat,\
                      'n_repeat':n_repeat,'n_samples':n_samples,'avg_neighbor':avg_neighbor,'ci_alpha':ci_alpha,'N':N, 'ci_test':ci_test, 'unequal_noise':unequal_noise}
        
        
    return res_repeat


### GENERATE SAMPLE RESULTS FOR equal_noise and unequal_noise 
# n_repeat = 500
# ci_alpha = 0.1

# res_8_equal_noise_10k = run_repeat(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=10_000,ci_alpha=ci_alpha,unequal_noise=False)
# res_8_equal_noise_100k = run_repeat(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=100_000,ci_alpha=ci_alpha,unequal_noise=False)

# res_8_unequal_noise_10k = run_repeat(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=10_000,ci_alpha=ci_alpha,unequal_noise=True)
# res_8_unequal_noise_100k = run_repeat(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=100_000,ci_alpha=ci_alpha,unequal_noise=True)

# res_8_equal_noise = {'res_8_equal_noise_10k':res_8_equal_noise_10k,\
#                     'res_8_equal_noise_100k':res_8_equal_noise_100k}

# res_8_unequal_noise = {'res_8_unequal_noise_10k':res_8_unequal_noise_10k,\
#                     'res_8_unequal_noise_100k':res_8_unequal_noise_100k}

# f = open('./results/skeleton_8_equal_noise.pkl','wb')
# pkl.dump(res_8_equal_noise,f)
# f.close()    

# f = open('./results/skeleton_8_unequal_noise.pkl','wb')
# pkl.dump(res_8_unequal_noise,f)
# f.close()    

#%%
# GENERATE SAMPLE RESULTS FOR VARYING MIXING RATE

# n_repeat = 500
# ci_alpha = 0.1
# n_samples = 1_000

# # n = 6
# res_6_mix_50 = run_repeat(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=n_samples,mix_rate=0.5,ci_alpha=ci_alpha)
# res_6_mix_90 = run_repeat(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=n_samples,mix_rate=0.9,ci_alpha=ci_alpha)

# res_6_mix = {'res_6_mix_50':res_6_mix_50,'res_6_mix_90':res_6_mix_90}

# f = open('./results/skeleton_6_mix_1k.pkl','wb')
# pkl.dump(res_6_mix,f)
# f.close()    


#%%
# # GENERATE SAMPLE RESULTS FOR COMPARING DIFFERENT CI TESTS

# n_repeat = 100
# ci_alpha = 0.1

# res_6_10k_partial = run_repeat(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=20000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='partial_correlation')
# res_6_10k_GCM = run_repeat(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=20000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='GCM')

# f = open('./results/skeleton_6_10k_partial.pkl','wb')
# pkl.dump(res_6_10k_partial,f)
# f.close()    

# f = open('./results/skeleton_6_10k_GCM.pkl','wb')
# pkl.dump(res_6_10k_GCM,f)
# f.close()    

