"""
run simulations for recovering orientations
"""
#%%
import warnings
warnings.simplefilter('ignore')
import os
import numpy as np
import causaldag as cd
from causaldag import partial_correlation_suffstat
from utils import generate_samples, is_forest, find_mixture_skeleton
from utils import create_true_mixture_graph_for_tree_DAGs, find_unshielded_triples
import pickle as pkl

if not os.path.exists('./results'):
    os.makedirs('./results')

def run_once_oriented(N=6,avg_neighbor=2,n_samples=5000,mix_rate=0.5,ci_alpha=0.1,ci_test='partial_correlation'):
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
    
    sepsets_double = {}
    for (i,j) in sepsets:
        sepsets_double[(i,j)] = sepsets[(i,j)]
        sepsets_double[(j,i)] = sepsets[(i,j)]
        
    # call ground truth
    res = create_true_mixture_graph_for_tree_DAGs(G1,G2) 
    GM_skel_GT = res['GM_skel']
    sepsets_GT = res['sepsets']
    EM_GT = res['EM']
    EU_GT = res['EU']
    EE_GT = res['EE']
    GM_oriented_GT = res['GM_oriented']
    GM_partially_oriented_GT = res['GM_partially_oriented']
    fully_oriented_edges_GT = res['fully_oriented_edges']
    partially_oriented_edges_GT = res['partially_oriented_edges']
    delta_oriented_edges_GT = res['delta_oriented_edges']
    
    #%
    
    GM_partially_oriented = GM_skel.copy()
    unshielded_triples_in_skel = find_unshielded_triples(GM_skel)
    
    for (i,k,j) in unshielded_triples_in_skel:
        if (i,j) in sepsets_double:
            if k not in sepsets_double[(i,j)]:
                # now (i,k,j) is a qualifying unshielded triple
                if ((i in non_delta) & (j in non_delta) & (k in delta)):
                    GM_partially_oriented[i,k] = 2
                    GM_partially_oriented[k,i] = 1
                    GM_partially_oriented[j,k] = 2
                    GM_partially_oriented[k,j] = 1   
                elif ((i in non_delta) & (j in delta) & (k in delta)):
                    GM_partially_oriented[i,k] = 2
                    GM_partially_oriented[k,i] = 1
                    GM_partially_oriented[j,k] = 4 # trivial
                    GM_partially_oriented[k,j] = 4 # trivial
                elif ((i in delta) & (j in non_delta) & (k in delta)):
                    GM_partially_oriented[i,k] = 4 # trivial
                    GM_partially_oriented[k,i] = 4 # trivial
                    GM_partially_oriented[j,k] = 2
                    GM_partially_oriented[k,j] = 1
                elif ((i in non_delta) & (j in non_delta) & (k in non_delta)):
                    GM_partially_oriented[i,k] = 2
                    GM_partially_oriented[k,i] = 3
                    GM_partially_oriented[j,k] = 2
                    GM_partially_oriented[k,j] = 3         
                elif ((i in non_delta) & (j in delta) & (k in non_delta)):
                    GM_partially_oriented[i,k] = 2
                    GM_partially_oriented[k,i] = 3
                elif ((i in delta) & (j in non_delta) & (k in non_delta)):
                    GM_partially_oriented[j,k] = 2
                    GM_partially_oriented[k,j] = 3
                    
    # well, if we have delta, direct delta-delta pairs regardless of unshielded triples
    for i in delta:
        for j in delta:
            if i!=j:
                GM_partially_oriented[i,j] = 4
                GM_partially_oriented[j,i] = 4
                
    fully_oriented_edges = [(i,j) for i in range(N) for j in range(N) if GM_partially_oriented[i,j]==2 and GM_partially_oriented[j,i]==1]
    partially_oriented_edges = [(i,j) for i in range(N) for j in range(N) if GM_partially_oriented[i,j]==2 and GM_partially_oriented[j,i]==3]
    delta_oriented_edges = [(i,j) for i in range(N) for j in range(N) if GM_partially_oriented[i,j]==4 and GM_partially_oriented[j,i]==4]
    
    fully_oriented_tp = len(set(fully_oriented_edges) & set(fully_oriented_edges_GT))
    fully_oriented_fp = len(set(fully_oriented_edges) - set(fully_oriented_edges_GT))
    fully_oriented_fn = len(set(fully_oriented_edges_GT) - set(fully_oriented_edges))
    partially_oriented_tp = len(set(partially_oriented_edges) & set(partially_oriented_edges_GT))
    partially_oriented_fp = len(set(partially_oriented_edges) - set(partially_oriented_edges_GT))
    partially_oriented_fn = len(set(partially_oriented_edges_GT) & set(partially_oriented_edges))
    
    oriented_tp = fully_oriented_tp + partially_oriented_tp
    oriented_fp = fully_oriented_fp + partially_oriented_fp
    oriented_fn = fully_oriented_fn + partially_oriented_fn

    return res, oriented_tp, oriented_fp, oriented_fn


def run_repeat_oriented(n_repeat=100,N=6,avg_neighbor=2,n_samples=5000,mix_rate=0.5,ci_alpha=0.1,ci_test='partial_correlation'):
    n_oriented_tp = 0
    n_oriented_fp = 0
    n_oriented_fn = 0
    for run_idx in range(n_repeat):
        res, oriented_tp, oriented_fp, oriented_fn = \
            run_once_oriented(N=N,avg_neighbor=avg_neighbor,n_samples=n_samples,mix_rate=mix_rate,ci_alpha=ci_alpha,ci_test=ci_test)
    
        n_oriented_tp += oriented_tp
        n_oriented_fp += oriented_fp
        n_oriented_fn += oriented_fn
        print('run_idx:',run_idx)

    oriented_precision = n_oriented_tp / (n_oriented_tp+n_oriented_fp)
    oriented_recall = n_oriented_tp / (n_oriented_tp+n_oriented_fn)
    oriented_f1 = 2 * oriented_precision * oriented_recall / (oriented_precision+oriented_recall)
    
    res_repeat = {'oriented_precision':oriented_precision, 'oriented_recall':oriented_recall, 'oriented_f1':oriented_f1,\
                  'n_oriented_tp':n_oriented_tp,'n_oriented_fp':n_oriented_fp,'n_oriented_fn':n_oriented_fn,\
                      'n_repeat':n_repeat,'n_samples':n_samples,'avg_neighbor':avg_neighbor,'ci_alpha':ci_alpha,'N':N}
    
    return res_repeat
    
#%%
n_repeat = 100
ci_alpha = 0.1

res_6_1k_partial = run_repeat_oriented(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=2000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='partial_correlation')
res_6_1k_GCM = run_repeat_oriented(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=2000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='GCM')

f = open('./results/orientation_6_1k_partial.pkl','wb')
pkl.dump(res_6_1k_partial,f)
f.close()    

f = open('./results/orientation_6_1k_GCM.pkl','wb')
pkl.dump(res_6_1k_GCM,f)
f.close()    

res_6_3k_partial = run_repeat_oriented(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=6000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='partial_correlation')
res_6_3k_GCM = run_repeat_oriented(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=6000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='GCM')

f = open('./results/orientation_6_3k_partial.pkl','wb')
pkl.dump(res_6_3k_partial,f)
f.close()    

f = open('./results/orientation_6_3k_GCM.pkl','wb')
pkl.dump(res_6_3k_GCM,f)
f.close()    

res_6_10k_partial = run_repeat_oriented(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=20000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='partial_correlation')
res_6_10k_GCM = run_repeat_oriented(n_repeat=n_repeat,N=6,avg_neighbor=2,n_samples=20000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='GCM')

f = open('./results/orientation_6_10k_partial.pkl','wb')
pkl.dump(res_6_10k_partial,f)
f.close()    

f = open('./results/orientation_6_10k_GCM.pkl','wb')
pkl.dump(res_6_10k_GCM,f)
f.close()    

res_8_1k_partial = run_repeat_oriented(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=2000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='partial_correlation')
res_8_1k_GCM = run_repeat_oriented(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=2000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='GCM')

f = open('./results/orientation_8_1k_partial.pkl','wb')
pkl.dump(res_8_1k_partial,f)
f.close()    

f = open('./results/orientation_8_1k_GCM.pkl','wb')
pkl.dump(res_8_1k_GCM,f)
f.close()    

res_8_3k_partial = run_repeat_oriented(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=6000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='partial_correlation')
res_8_3k_GCM = run_repeat_oriented(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=6000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='GCM')

f = open('./results/orientation_8_3k_partial.pkl','wb')
pkl.dump(res_8_3k_partial,f)
f.close()    

f = open('./results/orientation_8_3k_GCM.pkl','wb')
pkl.dump(res_8_3k_GCM,f)
f.close()    

res_8_10k_partial = run_repeat_oriented(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=20000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='partial_correlation')
res_8_10k_GCM = run_repeat_oriented(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=20000,mix_rate=0.5,ci_alpha=ci_alpha,ci_test='GCM')

f = open('./results/orientation_8_10k_partial.pkl','wb')
pkl.dump(res_8_10k_partial,f)
f.close()    

f = open('./results/orientation_8_10k_GCM.pkl','wb')
pkl.dump(res_8_10k_GCM,f)
f.close()    


#%%
# n_repeat = 100
# n_samples = 50_000

# res_mix_50 = run_repeat_oriented(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=n_samples,mix_rate=0.5,ci_alpha=0.1)

# res_mix_75 = run_repeat_oriented(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=n_samples,mix_rate=0.75,ci_alpha=0.1)

# res_mix_90 = run_repeat_oriented(n_repeat=n_repeat,N=8,avg_neighbor=2,n_samples=n_samples,mix_rate=0.9,ci_alpha=0.1)

# #%%

# res_6_1k = run_repeat_oriented(n_repeat=500,N=6,avg_neighbor=2,n_samples=1000,ci_alpha=0.1)
# res_6_3k = run_repeat_oriented(n_repeat=500,N=6,avg_neighbor=2,n_samples=3000,ci_alpha=0.1)
# res_6_10k = run_repeat_oriented(n_repeat=500,N=6,avg_neighbor=2,n_samples=10000,ci_alpha=0.1)
# res_6_30k = run_repeat_oriented(n_repeat=500,N=6,avg_neighbor=2,n_samples=30000,ci_alpha=0.1)
# res_6_50k = run_repeat_oriented(n_repeat=500,N=6,avg_neighbor=2,n_samples=50000,ci_alpha=0.1)
# res_6_100k = run_repeat_oriented(n_repeat=500,N=6,avg_neighbor=2,n_samples=100000,ci_alpha=0.1)

# #%%
# res_8_1k = run_repeat_oriented(n_repeat=500,N=8,avg_neighbor=2,n_samples=1000,ci_alpha=0.1)
# res_8_3k = run_repeat_oriented(n_repeat=500,N=8,avg_neighbor=2,n_samples=3000,ci_alpha=0.1)
# res_8_10k = run_repeat_oriented(n_repeat=500,N=8,avg_neighbor=2,n_samples=10000,ci_alpha=0.1)
# res_8_30k = run_repeat_oriented(n_repeat=500,N=8,avg_neighbor=2,n_samples=30000,ci_alpha=0.1)
# res_8_50k = run_repeat_oriented(n_repeat=500,N=8,avg_neighbor=2,n_samples=50000,ci_alpha=0.1)
# res_8_100k = run_repeat_oriented(n_repeat=500,N=8,avg_neighbor=2,n_samples=100000,ci_alpha=0.1)

# #%%
# res_6 = {'res_6_1k':res_6_1k, 'res_6_3k':res_6_3k, 'res_6_10k':res_6_10k, 'res_6_30k':res_6_30k,\
#          'res_6_50k':res_6_50k,'res_6_100k':res_6_100k}

# res_8 = {'res_8_1k':res_8_1k, 'res_8_3k':res_8_3k, 'res_8_10k':res_8_10k, 'res_8_30k':res_8_30k,\
#          'res_8_50k':res_8_50k,'res_8_100k':res_8_100k}



# f = open('./results/orientation_6.pkl','wb')
# pkl.dump(res_6,f)
# f.close()    

# f = open('./results/orientation_8.pkl','wb')
# pkl.dump(res_8,f)
# f.close()    
    
