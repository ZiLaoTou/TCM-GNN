import numpy as np
import torch
import dgl
from dgl.data.utils import save_graphs
import pandas as pd
import os
from tqdm import tqdm
def sparse_matrix(matrix, method='knn',k=10,thre=0.9):
    n = matrix.shape[0]
    sparsed_matrix = np.zeros((n, n))

    if method == 'knn':
        idx = np.argsort(matrix)[:, -1:-(k+1)]
        idx_arr = idx.reshape(1,-1)
        x = np.arange(matrix.shape[0])
        x = np.repeat(x,k)
        sparsed_matrix[x,idx_arr[0]] = matrix[x,idx_arr[0]]
        sparsed_matrix = (sparsed_matrix + sparsed_matrix.T)/2

        assert (sparsed_matrix == sparsed_matrix.T).all(), "The returned matrix is asymmetric !"

        return sparsed_matrix
    
    elif method == 'thre1':
        index = np.where(matrix>np.percentile(matrix, (1-thre)*100))
        sparsed_matrix[index[0],index[1]] = matrix[index[0],index[1]]
        sparsed_matrix = (sparsed_matrix + sparsed_matrix.T)/2

        assert (sparsed_matrix == sparsed_matrix.T).all(), "The returned matrix is asymmetric !"

        return sparsed_matrix
    
    elif method == 'thre2':
        
        abs_matrix = np.abs(matrix) 
        index = np.where(abs_matrix>np.percentile(abs_matrix, (1-thre)*100)) 
        sparsed_matrix[index[0], index[1]] = matrix[index[0], index[1]]
        sparsed_matrix = (sparsed_matrix + sparsed_matrix.T) / 2
                           
        max_min_index = np.argmax(abs_matrix, axis=1)
        max_min_values = matrix[np.arange(n), max_min_index]

        
        isolated_rows = np.sum(sparsed_matrix, axis=1) == 0
        sparsed_matrix[isolated_rows, max_min_index[isolated_rows]] = max_min_values[isolated_rows]
        sparsed_matrix[max_min_index[isolated_rows], isolated_rows] = max_min_values[isolated_rows]

        sparsed_matrix = (sparsed_matrix + sparsed_matrix.T) / 2

        assert (sparsed_matrix == sparsed_matrix.T).all(), "The returned matrix is asymmetric !"

        return torch.from_numpy(sparsed_matrix)

    elif method == 'thre3':
        
        matrix = np.abs(matrix) 
        index = np.where(matrix>np.percentile(matrix, (1-thre)*100)) 
        sparsed_matrix[index[0], index[1]] = matrix[index[0], index[1]]
        sparsed_matrix = (sparsed_matrix + sparsed_matrix.T) / 2
                           
        max_min_index = np.argmax(matrix, axis=1)
        max_min_values = matrix[np.arange(n), max_min_index]

        
        isolated_rows = np.sum(sparsed_matrix, axis=1) == 0
        sparsed_matrix[isolated_rows, max_min_index[isolated_rows]] = max_min_values[isolated_rows]
        sparsed_matrix[max_min_index[isolated_rows], isolated_rows] = max_min_values[isolated_rows]

        sparsed_matrix = (sparsed_matrix + sparsed_matrix.T) / 2

        assert (sparsed_matrix == sparsed_matrix.T).all(), "The returned matrix is asymmetric !"

        return sparsed_matrix


prefix = 'after'
atlas = 'bn'
if atlas == 'bn':
    num_node = 246
else:
    num_node = 166

clinical_file = ''
node_feat_file = ''
edge_feat_file = ''
seg_pool_file = ''
clinical_info = pd.read_csv(clinical_file)
pool_info = pd.read_csv(seg_pool_file)
id_list = clinical_info['ID'].to_numpy()
num_sub = len(id_list)

labels = torch.zeros((num_sub, )).float()
sites = torch.zeros((num_sub, )).long()
seg_pool_ids = []
graphs = []

for i, id_value in tqdm(enumerate(id_list)):
    labels[i] = clinical_info.loc[clinical_info['ID'] == id_value, '12month'].values[0]
    sites[i] = clinical_info.loc[clinical_info['ID'] == id_value, 'Cohort'].values[0]
    pool_id = pool_info[pool_info['ID'] == id_value].iloc[:, 1:].to_numpy()
    pool_id = torch.tensor(pool_id).flatten()
    seg_pool_ids.append(pool_id)
    node_feat_path_t1 = os.path.join(node_feat_file, id_value, prefix+'_'+atlas+'_t1_selected_normed_rfeature.npy')
    node_feat_path_t1c = os.path.join(node_feat_file, id_value, prefix+'_'+atlas+'_t1c_selected_normed_rfeature.npy')
    node_feat_path_t2 = os.path.join(node_feat_file, id_value, prefix+'_'+atlas+'_t2_selected_normed_rfeature.npy')
    node_feat_path_flair = os.path.join(node_feat_file, id_value, prefix+'_'+atlas+'_flair_selected_normed_rfeature.npy')

    edge_feat_path_t1 = os.path.join(edge_feat_file, id_value, prefix+'_'+atlas+'_t1_pc_matrix.npy')
    edge_feat_path_t1c = os.path.join(edge_feat_file, id_value, prefix+'_'+atlas+'_t1c_pc_matrix.npy')
    edge_feat_path_t2 = os.path.join(edge_feat_file, id_value, prefix+'_'+atlas+'_t2_pc_matrix.npy')
    edge_feat_path_flair = os.path.join(edge_feat_file, id_value, prefix+'_'+atlas+'_flair_pc_matrix.npy')

    node_feat_t1 = torch.from_numpy(np.load(node_feat_path_t1))
    node_feat_t1c = torch.from_numpy(np.load(node_feat_path_t1c))
    node_feat_t2 = torch.from_numpy(np.load(node_feat_path_t2))
    node_feat_flair = torch.from_numpy(np.load(node_feat_path_flair))

    node_feat = torch.cat([node_feat_t1, node_feat_t1c, node_feat_t2, node_feat_flair], dim=1)

    edge_feat_t1 = np.load(edge_feat_path_t1)
    edge_feat_t1c = np.load(edge_feat_path_t1c)
    edge_feat_t2 = np.load(edge_feat_path_t2)
    edge_feat_flair = np.load(edge_feat_path_flair)

    np.fill_diagonal(edge_feat_t1, 0)
    np.fill_diagonal(edge_feat_t1c, 0)
    np.fill_diagonal(edge_feat_t2, 0)
    np.fill_diagonal(edge_feat_flair, 0)

    edge_feat_t1 = np.arctanh(edge_feat_t1) 
    edge_feat_t1c = np.arctanh(edge_feat_t1c)
    edge_feat_t2 = np.arctanh(edge_feat_t2)
    edge_feat_flair = np.arctanh(edge_feat_flair)

    edge_feat_t1 = sparse_matrix(edge_feat_t1, method='thre2', thre=0.3)
    edge_feat_t1c = sparse_matrix(edge_feat_t1c, method='thre2', thre=0.3)
    edge_feat_t2 = sparse_matrix(edge_feat_t2, method='thre2', thre=0.3)
    edge_feat_flair = sparse_matrix(edge_feat_flair, method='thre2', thre=0.3)

    graph_data = {
                (atlas, 't1', atlas): (torch.where(edge_feat_t1!=0)[0], torch.where(edge_feat_t1!=0)[1]),
                (atlas, 't1c', atlas): (torch.where(edge_feat_t1c!=0)[0], torch.where(edge_feat_t1c!=0)[1]),
                (atlas, 't2', atlas): (torch.where(edge_feat_t2!=0)[0], torch.where(edge_feat_t2!=0)[1]),
                (atlas, 'flair', atlas): (torch.where(edge_feat_flair!=0)[0], torch.where(edge_feat_flair!=0)[1]),
    }
    node_info = {atlas: num_node}
    g = dgl.heterograph(data_dict=graph_data, num_nodes_dict=node_info)
    g.ndata['feat'] = node_feat
    graphs.append(g)

seg_pool_ids = torch.stack(seg_pool_ids).float()
graph_lables = {'labels': labels, 'pool_ids': seg_pool_ids, 'sites': sites}


save_graphs("", g_list=graphs, labels=graph_lables)






