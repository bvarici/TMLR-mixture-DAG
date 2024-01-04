import numpy as np
from numpy.random import normal
import numpy.linalg as LA
from numpy.linalg import inv

import causaldag as cd
from causaldag import partial_correlation_suffstat, partial_correlation_test
from scipy import stats
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
import itertools

def GCM(X, Y, Z):
    if Z is None:
        Z = np.zeros(X.shape)

    if len(Z.shape) == 1:
        Z = np.expand_dims(Z, axis = 1)
    # Compute residuals of X|Z. I chose Lasso as estimator. 
    model_X = LassoCV(cv = 5).fit(Z, X)
    res_X = X - model_X.predict(Z)
    # and residuals of Y|Z.
    model_Y = LassoCV(cv = 5).fit(Z, Y)
    res_Y = Y - model_Y.predict(Z)

    nn = res_X.shape[0]
    R = np.multiply(res_X, res_Y)
    R_sq = R ** 2
    meanR = np.mean(R)
    test_stat = np.sqrt(nn) * meanR / np.sqrt(np.mean(R_sq) - meanR ** 2)
    pval = 2 * (1 - stats.norm.cdf(np.abs(test_stat)))
    return pval


def generate_samples(B,means,variances,n_samples,topological_order):
    '''
    Parameters
    ----------
    B : 
        autoregression weight matrix. assumed to be strictly upper triangular
    means : 
        internal noise means.
    variances : 
        internal noise variances.
    n_samples : integer
        number of samples to generate
    topological_order: list
        for proper sampling, we need the topological order

    Returns
    -------
    samples : n_samples x p matrix
        DAG samples

    ''' 
    p = len(means)
    samples = np.zeros((n_samples,p))
    noise = np.zeros((n_samples,p))
    for ix, (mean,var) in enumerate(zip(means,variances)):
        noise[:,ix] = np.random.normal(loc=mean,scale=var ** .5, size=n_samples)
        
    for node in topological_order:
        parents_node = np.where(B[:,node])[0]
        if len(parents_node)!=0:
            parents_vals = samples[:,parents_node]
            samples[:,node] = np.sum(parents_vals * B[parents_node,node],axis=1) + noise[:,node]
        else:
            samples[:,node] = noise[:,node]
            
    return samples

def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def allsubsets(s):
    N = len(s)
    temp = []
    for n in range(0,N+1):
        temp.append(list(itertools.combinations(s, n)))

    ss = []
    for n in range(0,N+1):
        ss += temp[n]

    ss = [list(a) for a in ss]
    return ss

def find_non_invariant_coordinates(*matrices):
    # Create an array to store the differences
    diffs = [matrices[i] != matrices[j] for i in range(len(matrices)) for j in range(i + 1, len(matrices))]
    
    # Combine the differences using logical OR
    non_invariant_coords = np.logical_or.reduce(diffs)
    
    # Get the coordinates where differences exist
    coords = np.transpose(np.nonzero(non_invariant_coords))
    
    return coords


def create_composite_DAG(*G_all):
    m = len(G_all)
    A_all = [G_all[i].to_amat()[0] for i in range(m)]
    changing_coords = find_non_invariant_coordinates(*A_all)
    delta = np.array(list(set(changing_coords[:,1])))

    N = G_all[0].nnodes

    #build the large composite DAG to test CIs: y is the node 2N; G1: 0 to N-1; G2: N to 2N-1'
    GC_mat = np.zeros((m*N+1,m*N+1))

    for i in range(m):
        GC_mat[-1,delta+i*N] = 1
        GC_mat[i*N:(i+1)*N, i*N:(i+1)*N] = A_all[i]

    GC = cd.DAG.from_amat(GC_mat)
    return GC

def dsep_composite_DAG(GC,N,i,j,S=[]):
    # check d-sep in large composite DAG
    # for simplicity, just take N as input
    #N = int((G.nnodes-1)/2)
    i_bar = [i,i+N]
    j_bar = [j,j+N]
    if len(S) > 0:
        S_bar = np.concatenate([np.array(S),np.array(S)+N])
        return GC.dsep(i_bar,j_bar,S_bar)
    else:
        return GC.dsep(i_bar,j_bar)
    
def find_unshielded_triples(adjacency_matrix):
    unshielded_triples = []
    num_nodes = len(adjacency_matrix)
    
    for node in range(num_nodes):
        for u in range(num_nodes):
            for v in range(num_nodes):
                if u != v and adjacency_matrix[u][v] == 0 and adjacency_matrix[node][u] == 1 and adjacency_matrix[node][v] == 1:
                    unshielded_triples.append((u,node, v))
    
    return unshielded_triples
    
  
def create_true_skeleton_mixture_graph(G1,G2):
    N = G1.nnodes
    A1 = G1.to_amat()[0]
    A2 = G2.to_amat()[0]
    E1 = [tuple(a) for a in np.argwhere(A1!=0)]
    E2 = [tuple(a) for a in np.argwhere(A2!=0)]
    Eshared = [edge for edge in E1 if edge in E2]
    delta = np.where(np.sum(A1!=A2,0))[0]
    non_delta = [i for i in range(N) if i not in delta]
    #all_pairs = [(i,j) for i in range(N) for j in range(N) if i < j] 
    # union graph
    U_mat = A1 + A1.T +  A2 + A2.T
    # union edges 
    EU = [tuple(a) for a in np.argwhere(U_mat!=0) if a[0] < a[1]]
    # potential emergent edges
    potential_EE = [tuple(a) for a in np.argwhere(U_mat==0) if a[0] < a[1]]
    EE = [tuple(a) for a in np.argwhere(U_mat==0) if a[0] < a[1]]
    
    # build the large "composite DAG": y is the node 0; G1: 1 to N; G2: N+1 to 2N'
    GC_mat = np.zeros((2*N+1,2*N+1))
    GC_mat[-1,delta] = 1
    GC_mat[-1,delta+N] = 1
    GC_mat[:N,:N] = A1
    GC_mat[N:-1,N:-1] = A2
    GC = cd.DAG.from_amat(GC_mat)
    sepsets = {}

    
    for (i,j) in potential_EE:
        rest_nodes = [k for k in range(N) if k not in (i,j)]
        all_S = allsubsets(rest_nodes)
        for S in all_S:
            sep_flag = dsep_composite_DAG(GC,N,i,j,S)
            #print((i,j),S,sep_flag)
            if sep_flag == True:
                sepsets[(i,j)] = S
                EE.remove((i,j))
                break
    
    GM_skel = np.zeros((N,N))
    GM_skel[np.where(U_mat)] = 1
    for (i,j) in EE:
        GM_skel[i,j] = 1
        GM_skel[j,i] = 1    
           
    EE_d_d = [(i,j) for i in delta for j in delta if i < j if (i,j) in EE]
    EE_non_d_d = [(i,j) for i in non_delta for j in delta if i < j if (i,j) in EE]
    EE_non_d_non_d = [(i,j) for i in non_delta for j in non_delta if i < j if (i,j) in EE]
    
    EM = EU + EE

    res = {'GC':GC, 'GM_skel':GM_skel, 'sepsets':sepsets, 'E1':E1, 'E2': E2, 'EM': EM, 'EU':EU, 'EE':EE, \
           'EE_d_d':EE_d_d, 'EE_non_d_d':EE_non_d_d, 'EE_non_d_non_d':EE_non_d_non_d,'delta':delta}
        
    return res

def create_true_mixture_graph_for_tree_DAGs(*G_all):
    m = len(G_all)
    N = G_all[0].nnodes
    A_all = [G_all[i].to_amat()[0] for i in range(m)]
    E_all = [[tuple(a) for a in np.argwhere(A_all[i]!=0)] for i in range(m)]

    changing_coords = find_non_invariant_coordinates(*A_all)
    delta = np.array(list(set(changing_coords[:,1])))
    non_delta = [i for i in range(N) if i not in delta]

    # union graph
    U_mat = np.zeros((N,N))
    for i in range(m):
        U_mat += (A_all[i] + A_all[i].T)

    Eshared = list(set.intersection(*map(set, E_all)))

    # union edges 
    EU = [tuple(a) for a in np.argwhere(U_mat!=0) if a[0] < a[1]]
    # potential emergent edges
    potential_EE = [tuple(a) for a in np.argwhere(U_mat==0) if a[0] < a[1]]
    EE = [tuple(a) for a in np.argwhere(U_mat==0) if a[0] < a[1]]
    
    # build the large "composite DAG": y is the node 0; G1: 1 to N; G2: N+1 to 2N'
    GC = create_composite_DAG(*G_all)
    sepsets = {}
    
    for (i,j) in potential_EE:
        rest_nodes = [k for k in range(N) if k not in (i,j)]
        all_S = allsubsets(rest_nodes)
        for S in all_S:
            sep_flag = dsep_composite_DAG(GC,N,i,j,S)
            #print((i,j),S,sep_flag)
            if sep_flag == True:
                sepsets[(i,j)] = S
                EE.remove((i,j))
                break
            
    sepsets_double = {}
    for (i,j) in sepsets:
        sepsets_double[(i,j)] = sepsets[(i,j)]
        sepsets_double[(j,i)] = sepsets[(i,j)]
    
    GM_skel = np.zeros((N,N))
    GM_skel[np.where(U_mat)] = 1
    for (i,j) in EE:
        GM_skel[i,j] = 1
        GM_skel[j,i] = 1    
           
    EE_d_d = [(i,j) for i in delta for j in delta if i < j if (i,j) in EE]
    EE_non_d_d = [(i,j) for i in non_delta for j in delta if i < j if (i,j) in EE]
    EE_non_d_non_d = [(i,j) for i in non_delta for j in non_delta if i < j if (i,j) in EE]
    
    EM = EU + EE

    'fully orient edges '
    GM_oriented = np.zeros((N,N))

    # -->> edges: onto j in non_delta
    for (i,j) in Eshared:
        if j in non_delta:
            GM_oriented[i,j] = 2
            GM_oriented[j,i] = 1

    # -->> edges: mixture edges that point towards j in delta 
    for (i,j) in EE_non_d_d:
        if j in delta:
            GM_oriented[i,j] = 2
            GM_oriented[j,i] = 1
        elif i in delta:
            GM_oriented[j,i] = 2
            GM_oriented[i,j] = 1
            
    for (i,j) in EU:
        if j in delta:
            GM_oriented[i,j] = 2
            GM_oriented[j,i] = 1
        elif i in delta:
            GM_oriented[j,i] = 2
            GM_oriented[i,j] = 1

    # <<-->> edges: emergent edges between i and j in non_delta
    for (i,j) in EE_non_d_non_d:
        GM_oriented[i,j] = 2
        GM_oriented[j,i] = 2

    # <<-->> edges: mixture edges between i and j in delta.    
    for i in delta:
        for j in delta:
            if i!=j:
                GM_oriented[i,j] = 2
                
                
    'partially orient edges, that is, what can we learn if delta is known'
    # 3: joker.  2: >>  1: --
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
                    GM_partially_oriented[j,k] = 4
                    GM_partially_oriented[k,j] = 4   
                elif ((i in delta) & (j in non_delta) & (k in delta)):
                    GM_partially_oriented[i,k] = 4
                    GM_partially_oriented[k,i] = 4
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


    res = {'GC':GC, 'GM_skel':GM_skel, 'GM_oriented':GM_oriented, 'GM_partially_oriented':GM_partially_oriented,\
           'fully_oriented_edges':fully_oriented_edges,\
               'partially_oriented_edges':partially_oriented_edges,\
                   'delta_oriented_edges':delta_oriented_edges,\
                       'sepsets':sepsets, 'EM': EM, 'EU':EU, 'EE':EE, \
           'EE_d_d':EE_d_d, 'EE_non_d_d':EE_non_d_d, 'EE_non_d_non_d':EE_non_d_non_d,'delta':delta}
        
    return res
    

def find_all_paths(adj_matrix, start_node, end_node):
    paths = []
    visited = [False] * len(adj_matrix)
    current_path = []

    def dfs(node):
        visited[node] = True
        current_path.append(node)

        if node == end_node:
            paths.append(current_path[:])  # Append a copy of the current path

        else:
            for neighbor in range(len(adj_matrix[node])):
                if adj_matrix[node][neighbor] and not visited[neighbor]:
                    dfs(neighbor)

        current_path.pop()
        visited[node] = False

    dfs(start_node)
    return paths

def is_forest(adj_matrix):
    N = len(adj_matrix)
    for i in range(N):
        for j in range(N):
            if len(find_all_paths(adj_matrix,i,j)) > 1:
                return False
            
    return True


def find_mixture_skeleton(datamix,ci_alpha=0.05,ci_test='partial_correlation'):
    N = datamix.shape[-1]
    sepsets = {}
    all_pairs = [(i,j) for i in range(N) for j in range(N) if i < j]
    all_edges = [(i,j) for i in range(N) for j in range(N) if i < j]
    A = np.ones((N,N))
    np.fill_diagonal(A,0)
    
    if ci_test == 'partial_correlation':
        suffstat = partial_correlation_suffstat(datamix)

    for (i,j) in all_pairs:
        rest_nodes = [k for k in range(N) if k not in (i,j)]
        all_S = allsubsets(rest_nodes)
        for S in all_S:
            if len(S) > 0:
                if ci_test == 'partial_correlation':
                    p_val = partial_correlation_test(suffstat,i,j,cond_set=S,alpha=ci_alpha)['p_value']
                elif ci_test == 'GCM':
                    p_val = GCM(datamix[:,i],datamix[:,j],datamix[:,S])            
            else:
                if ci_test == 'partial_correlation':
                    p_val = partial_correlation_test(suffstat,i,j,cond_set=(),alpha=ci_alpha)['p_value']
                elif ci_test == 'GCM':
                    p_val = GCM(datamix[:,i],datamix[:,j],None)            
           
            if p_val > ci_alpha:
                sepsets[(i,j)] = S   
                all_edges.remove((i,j))
                #print((i,j),S,p_val)
                A[i,j] = 0
                A[j,i] = 0
                break
        
    return A, all_edges, sepsets

