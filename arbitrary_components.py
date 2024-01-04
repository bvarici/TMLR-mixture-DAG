"""

Use CI tests to learn the skeleton.
Use partial correlation as CI test.

"""
#%%
import warnings
warnings.simplefilter('ignore')
import numpy as np
import os
import causaldag as cd
from utils import generate_samples, is_forest, find_mixture_skeleton, find_non_invariant_coordinates, create_true_mixture_graph_for_tree_DAGs
import pickle as pkl

if not os.path.exists('./results'):
    os.makedirs('./results')

def run_once(N=6,m=2,avg_neighbor=2,n_samples=5000,ci_alpha=0.1,ci_test='partial_correlation'):

    forest_flags = [False for _ in range(m)]
    A_all = np.zeros((m,N,N))

    # generate m forests
    for i in range(m):
        while forest_flags[i] is False:
            Ai = np.triu(np.random.uniform(size=[N,N])<(avg_neighbor/N),k=1)
            idx = np.random.permutation(N)
            Ai = Ai[idx][:,idx]
            if is_forest(Ai) is True:
                forest_flags[i] = True
                A_all[i] = Ai

    G_all = [cd.DAG.from_amat(A_all[i]) for i in range(m)]
    to_all = [G_all[i].topological_sort() for i in range(m)]

    # assign weights to the edges
    rand_weights = np.random.uniform(-2,-0.25,[N,N])* np.random.choice([-1,1],size=[N,N])
    W_all = [A_all[i]*rand_weights for i in range(m)]
    changing_coords = find_non_invariant_coordinates(*A_all)
    delta = list(set(changing_coords[:,1]))
    non_delta = [i for i in range(N) if i not in delta]
    mu_epsilon = np.random.uniform(-2,2,size=N)

    sigma_epsilon_all = [np.ones(N) for _ in range(m)]

    #% generate samples
    data_all = [generate_samples(W_all[i],mu_epsilon,sigma_epsilon_all[i],n_samples,topological_order=to_all[i]) for i in range(m)]
    datamix = np.vstack(data_all)
    np.random.shuffle(datamix)

    # call ground truth
    # GC = create_composite_DAG(*G_all)
    res = create_true_mixture_graph_for_tree_DAGs(*G_all) 
    GM_skel_GT = res['GM_skel']
    sepsets_GT = res['sepsets']
    EM_GT = res['EM']
    EU_GT = res['EU']
    EE_GT = res['EE']

    # call our algo
    GM_skel, EM, sepsets = find_mixture_skeleton(datamix,ci_alpha,ci_test=ci_test)

    sepsets_double = {}
    for (i,j) in sepsets:
        sepsets_double[(i,j)] = sepsets[(i,j)]
        sepsets_double[(j,i)] = sepsets[(i,j)]


    # overall
    skel_tp = len(set(EM_GT) & set(EM))
    skel_fp = len(set(EM)-set(EM_GT))
    skel_fn = len(set(EM_GT)-set(EM))
    # for union edges
    skel_u_tp = len(set(EU_GT) & set(EM))
    skel_u_fn = len(set(EU_GT)-set(EM))
    # for emergent edges
    skel_e_tp = len(set(EE_GT) & set(EM))
    skel_e_fn = len(set(EE_GT)-set(EM))

    return res, skel_tp, skel_fp, skel_fn, skel_u_tp, skel_u_fn, skel_e_tp, skel_e_fn

def run_repeat(n_repeat=10,N=6,m=2,avg_neighbor=2,n_samples=5000,ci_alpha=0.1,ci_test='partial_correlation'):
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
                    run_once(N=N,m=m,avg_neighbor=avg_neighbor,n_samples=n_samples,ci_alpha=ci_alpha,ci_test=ci_test)
            
    
        n_delta += len(res['delta'])
        n_skel_tp += skel_tp
        n_skel_fp += skel_fp
        n_skel_fn += skel_fn
        n_skel_u_tp += skel_u_tp
        n_skel_u_fn += skel_u_fn
        n_skel_e_tp += skel_e_tp
        n_skel_e_fn += skel_e_fn

        print('N=',N,' m=',m,' n_samples=',n_samples,' run_idx:',run_idx)
    
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

    if n_skel_tn + n_skel_fn > 0:
        separable_precision = n_skel_tn / (n_skel_tn + n_skel_fn)
    else: 
        separable_precision = None

    if n_skel_tn + n_skel_fp > 0:   
        separable_recall = n_skel_tn / (n_skel_tn + n_skel_fp)
    else:
        separable_recall = None

    if separable_precision is None or separable_recall is None:
        separable_f1 = None   
    else:
        separable_f1 = 2*separable_precision*separable_recall / (separable_precision+separable_recall)
    
    res_repeat = {'skel_precision':skel_precision, 'skel_recall':skel_recall, 'skel_f1':skel_f1,\
                  'skel_u_recall':skel_u_recall,'skel_e_recall':skel_e_recall, \
                      'separable_precision':separable_precision, 'separable_recall':separable_recall,\
                          'separable_f1':separable_f1,'avg_delta':n_delta/n_repeat,\
                      'n_repeat':n_repeat,'n_samples':n_samples,'avg_neighbor':avg_neighbor,'ci_alpha':ci_alpha,'N':N, 'ci_test':ci_test}
        
        
    return res_repeat


#%% GENERATING SAMPLE RESULTS

N=6
avg_neighbor=2
ci_alpha = 0.1

n_repeat = 500

res_6_10k_m2 = run_repeat(n_repeat=n_repeat,N=6,m=2,avg_neighbor=2,n_samples=10_000,ci_alpha=ci_alpha)
res_6_10k_m3 = run_repeat(n_repeat=n_repeat,N=6,m=3,avg_neighbor=2,n_samples=10_000,ci_alpha=ci_alpha)
res_6_10k_m4 = run_repeat(n_repeat=n_repeat,N=6,m=4,avg_neighbor=2,n_samples=10_000,ci_alpha=ci_alpha)
res_6_10k_m5 = run_repeat(n_repeat=n_repeat,N=6,m=5,avg_neighbor=2,n_samples=10_000,ci_alpha=ci_alpha)

res_6_arbitrary = {'res_6_10k_m2':res_6_10k_m2, 'res_6_10k_m3':res_6_10k_m3, 'res_6_10k_m4':res_6_10k_m4, 'res_6_10k_m5':res_6_10k_m5}

f = open('./results/skeleton_6_arbitrary.pkl','wb')
pkl.dump(res_6_arbitrary,f)
f.close()    
