import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import pandas as pd
import scanpy as sc
import os
import math
from collections import Counter
import random
from .utils import NetEncoder, NetDecoder, NetClassifier, label_transform_num, num_to_label
from scipy.sparse import issparse
from sklearn.preprocessing import scale, minmax_scale
from sklearn.mixture import GaussianMixture
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2.robjects.conversion import localconverter
import giniclust3 as gc

def preprocess(source_dataset, target_dataset, take_log = True, standardization=True, scaling=True):

    if take_log:
        source_dataset = (source_dataset+1).apply(np.log2)
        target_dataset = (target_dataset+1).apply(np.log2)
    if standardization:
        source_dataset = source_dataset.apply(scale, axis=0)
        target_dataset = target_dataset.apply(scale, axis=0)
    if scaling:
        source_dataset = minmax_scale(source_dataset, feature_range=(0, 1), axis=0)
        target_dataset = minmax_scale(target_dataset, feature_range=(0, 1), axis=0)
    
    source_dataset = np.array(source_dataset)
    target_dataset = np.array(target_dataset)
        
    return source_dataset, target_dataset


def select_gene(epr_s, label_s, feature):
     	
    n=len(epr_s)
    ro.r.source("/home/xiongyx/xyx/time_memory/mtANN/featureSelection.R")
    
    r_list = r.list()
    for i in range(n):
        print("Convert {}-th ref to R object".format(i))
        with localconverter(ro.default_converter + pandas2ri.converter): 
            r_matrix = ro.conversion.py2rpy(epr_s[i].T)
        r_list.rx2[f'sample{i+1}_name'] = r_matrix
    
    r_list_label = r.list()
    for i in range(n):
        r_label = ro.StrVector(label_s[i])
        r_list_label.rx2[f'sample{i+1}_name'] = r_label
        
    r_t_gene = ro.StrVector(feature)
    
    gene_result = ro.r.get_result(r_list, r_list_label, r_t_gene)
    py_list=[]
    for res in range(n):
        py_list.append(list(gene_result[res]))
    
    for s in range(n):
        print("Selecting genes with gc")
        adataRaw = sc.AnnData(epr_s[s])
        raw_genes = adataRaw.var.index
        sc.pp.normalize_per_cell(adataRaw, counts_per_cell_after=1e4)
        gc.gini.calGini(adataRaw, selection='p_value', p_value=0.01, min_gini_value=0)
        res_gene = adataRaw.var['gini']
        result = set(raw_genes[np.where(res_gene == True)[0]]).intersection(set(feature))
        py_list[s].append(np.array(list(result)))

    return(py_list)

def train_ae(encoder, decoder, data_loader_s, data_loader_t, lr=1e-2, epoch=30):

    num_iter = max(len(data_loader_s), len(data_loader_t))
    
    encoder.train()
    decoder.train()
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
    )
    loss_mse_func = nn.MSELoss()
    
    for ep in range(epoch):
        for iteration in range(num_iter):

            if (iteration % len(data_loader_s) == 0):
                iter_dataloader_s = iter(data_loader_s)
            
            if (iteration % len(data_loader_t) == 0):
                iter_dataloader_t = iter(data_loader_t)
                
            x_s, _ = iter_dataloader_s.next()
            x_t, = iter_dataloader_t.next()

            if x_s.is_sparse:
                x_s = x_s.to_dense()
            if x_t.is_sparse:
                x_t = x_t.to_dense()
            
            optimizer.zero_grad()
            
            feature_s = encoder(x_s)
            reconstruct_s = decoder(feature_s)

            feature_t = encoder(x_t)
            reconstruct_t = decoder(feature_t)
            
            loss = loss_mse_func(reconstruct_s, x_s) + loss_mse_func(reconstruct_t, x_t)
            
            loss.backward()
            optimizer.step()
            
    return encoder, decoder



def train_s(encoder, decoder, classifier, data_loader_s, data_loader_t, lr=1e-2, epoch=30):

    encoder.train()
    decoder.train()
    classifier.train()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters()),
        lr = lr,
    )
    loss_crossentropy_func = nn.CrossEntropyLoss()
    loss_mse_func = nn.MSELoss()
    
    for ep in range(epoch):
        for step, (x_s, y_s) in enumerate(data_loader_s):

            if (step % len(data_loader_t) == 0):
                iter_dataloader_t = iter(data_loader_t)

            x_t, = iter_dataloader_t.next()

            if x_s.is_sparse:
                x_s = x_s.to_dense()
            if x_t.is_sparse:
                x_t = x_t.to_dense()

            optimizer.zero_grad()

            feature_s = encoder(x_s)
            reconstruct_s = decoder(feature_s)
            prediction_s = classifier(feature_s)

            feature_t = encoder(x_t)
            reconstruct_t = decoder(feature_t)

            loss_cls = loss_crossentropy_func(prediction_s, y_s)
            loss_rec = loss_mse_func(reconstruct_s, x_s) + loss_mse_func(reconstruct_t, x_t)

            loss = 1 * loss_cls + 1 * loss_rec
                        
            loss.backward()
            optimizer.step()

        
    return encoder, decoder, classifier

def unseen_metric_cal(H_single, overall_P, hard_vote_label, ct_counts):

    # calculate intra-model measurement m_1
    en = pd.DataFrame(np.array(list(H_single.values())))
    en.iloc[np.where(np.isnan(en))] = 0
    m_1 = np.mean(np.array(en), axis = 0)
    m_1 = m_1.reshape(len(m_1), 1)

    # calculate inter-model measurement m_2    
    m_2 = []
    for l in range(overall_P.shape[0]):
        a = overall_P.iloc[l,np.where(overall_P.iloc[l,:] != 0)[0]].values
        a = a/np.sum(a)
        m_2.append((-a*np.log2(a)).sum()/np.log2(overall_P.shape[1]))
    m_2 = np.array(m_2).reshape(len(m_2), 1)
        
    # calculate inter-model measurement m_3
    m_3 = []
    for j in range(hard_vote_label.shape[0]):
        keys = []
        values = []
        for k, v in Counter(hard_vote_label.iloc[j,:]).items():
            keys.append(k)
            values.append(v)
        count = np.array(values)
        ct = np.array(keys)
        count = count / np.array([ct_counts.loc[ct_counts['ct'] == tt]['num'] for tt in ct])
        p_count = count/sum(count)
        m_3.append((-p_count*np.log2(p_count)).sum()/np.log2(ct_counts.shape[0]))
    m_3 = np.array(m_3).reshape(len(m_3), 1)
    
    m_1 = (m_1 - np.min(m_1))/(np.max(m_1) - np.min(m_1))
    m_2 = (m_2 - np.min(m_2))/(np.max(m_2) - np.min(m_2))
    m_3 = (m_3 - np.min(m_3))/(np.max(m_3) - np.min(m_3))
    m = (m_1+m_2+m_3)/3

    return(m)

def threshold_selection(m):

    d = [1,2,3,4,5]
    aic = []
    for dd in d:
        gm = GaussianMixture(n_components=dd, random_state=0).fit(m)
        aic.append(gm.aic(m))

    if (aic.index(np.min(aic))+1 == 1):
        ind_pred_df = []
    else:
        gm = GaussianMixture(n_components=(aic.index(np.min(aic))+1), random_state=0).fit(m)
        group = gm.predict(m)
        score = [np.mean(m[np.where(group == g)]) for g in np.unique(group)]

        if (len(np.where(np.array(score) >= 0.6)[0]) <= 1):
            ind_pred_df = np.where(group == score.index(np.max(score)))[0]
        else:
            gg = np.where(np.array(score) >= 0.6)[0]
            ind_pred_df = np.concatenate(([np.where(group == i)[0] for i in gg]))

    if len(ind_pred_df) != 0:
        threshold = np.min(m[ind_pred_df])
    else:
        threshold = 0

    return(ind_pred_df, threshold)

def mtANN(expression_s, label_s, expression_t, threshold="default", gene_select="default", CUDA=False):

# 	'''
# 	Input:
# 	:expression_s: A list of gene expression matrices for reference datasets, each matrix is formatted as a data frame where rowa are cells and columns are genes.
# 	:label_s: A list of cell-type labels corresponding to references in expression_s.
# 	:expression_t: The gene expression matrix of target dataset whose format is the same as reference datasets.
# 	:threshold: Either be default or a number between 0~1. This parameter indicates the threshold for unseen cell-type identification is selected using the method's default threshold or user-defined.
# 	:gene_select: "default" means that the default eight gene selection methods are used. Other values indicate that no gene selection is performed.
# 	Output:
# 	pred_label: 1D-array, metaphase annotations of target dataset.
# 	final_annotation: 1D-array, final annotation including unseen cell-type identification.
# 	m: 1D-array, metric for unseen cell-type identification.
# 	threshold_df: int, the threshold selected by mtANN.
# 	'''

    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
    celltype = list()
    for i in range(len(label_s)):
        celltype.extend(np.unique(label_s[i]))
    celltype = np.array(sorted(np.unique(celltype)))
        
    # Gene selection
    exp_rs = list()
    exp_t = list()
    rs_label = list()
    if gene_select == "default":
        print("Selecting genes with default methods")
        gene_set = select_gene(expression_s, label_s, np.array(expression_t.columns))
        
        for ref in range(len(expression_s)):
            for s in range(8):
                a = expression_s[ref][np.array(list(gene_set[ref][s]))]
                b = expression_t[np.array(list(gene_set[ref][s]))]
                data_s, data_t = preprocess(a, b)
                exp_rs.append(data_s)
                exp_t.append(data_t)
                rs_label.append(label_s[ref])
    else:
        for ref in range(len(expression_s)):
            gene_set = set(expression_s[ref].columns).intersection(set(expression_t.columns))
            a = expression_s[ref][np.array(list(gene_set))]
            b = expression_t[np.array(list(gene_set))]
            data_s, data_t = preprocess(a, b)
            exp_rs.append(data_s)
            exp_t.append(data_t)
            rs_label.append(label_s[ref])

        
    ct_in_data_counts = []
    for i in range(len(celltype)):
        a=[]
        for j in range(len(rs_label)):
            a.append((rs_label[j] == celltype[i]).any() * 1)
        ct_in_data_counts.append(np.sum(np.array(a)))
    ct_counts = pd.DataFrame(zip(celltype, ct_in_data_counts))
    ct_counts.columns = ["ct","num"]

    ns = len(exp_rs)
    num_t = exp_t[0].shape[0]
    
    # Base classification model training
    classifier_s = {}
    encoder = {}
    label_t_pred = {}
    y_t_pred = {}
    y_t_pred_softmax = {}
    H_single = {}
    cell_type_single = {}
    single_p = np.zeros(shape = (num_t, ns))
    for i in range(ns):
        print("training {}-th classification model".format(i+1))
        num_label_s, num_label_transform, _ = label_transform_num(rs_label[i])
        num_label = len(set(rs_label[i]))
        num_gene = exp_rs[i].shape[1]

        batch_size_s = int(math.ceil(max(Counter(rs_label[i]).values())/10))
        batch_size_t = int(np.min([math.ceil(num_t/10),600]))
        
        x_s = torch.from_numpy(exp_rs[i]).float()
        y_s = torch.from_numpy(num_label_s).long()
        x_t = torch.from_numpy(exp_t[i]).float()
            

        if CUDA:
            x_s = x_s.cuda()
            x_t = x_t.cuda()
            y_s = y_s.cuda()

        x_y_s = Data.TensorDataset(x_s, y_s)    
        dataloader_s = Data.DataLoader(
            dataset=x_y_s,
            batch_size=batch_size_s,
            shuffle=True,
            drop_last=True
        )

        x_y_t = Data.TensorDataset(x_t)
        dataloader_t = Data.DataLoader(
            dataset=x_y_t,
            batch_size=batch_size_t,
            shuffle=True,
            drop_last=True
        )

        encoder[i] = NetEncoder(input_dim=num_gene)
        decoder = NetDecoder(output_dim=num_gene)
        classifier_s[i] = NetClassifier(output_dim=num_label)

        if CUDA:
            encoder[i] = encoder[i].cuda()
            decoder = decoder.cuda()
            classifier_s[i] = classifier_s[i].cuda()
        
        encoder[i], decoder = train_ae(encoder[i], decoder, dataloader_s, dataloader_t)
        encoder[i], decoder, classifier_s[i] = train_s(encoder[i], decoder, classifier_s[i], dataloader_s, dataloader_t)

        encoder[i].eval()
        classifier_s[i].eval()
        with torch.no_grad():
        
            y_t_pred_softmax[i] = F.softmax(classifier_s[i](encoder[i](x_t)), dim=1).detach().cpu().numpy()
            y_t_pred[i] = classifier_s[i](encoder[i](x_t)).detach().max(1)[1].cpu().numpy()
            label_t_pred[i] = num_to_label(y_t_pred[i], num_label_transform)

            single_p[:,i] = np.max(y_t_pred_softmax[i], axis = 1)

            h = np.zeros(num_t)
            for c in range(num_t):
                a = y_t_pred_softmax[i][c,:][np.where(y_t_pred_softmax[i][c,:] != 0)[0]]
                h[c] = (-a*np.log2(a)).sum()/np.log2(len(a))
            H_single[i] = h
            cell_type_single[i] = num_to_label(np.unique(num_label_s), num_label_transform)        

    label_t_pred_mtx = pd.DataFrame(label_t_pred)

    overall_P = np.zeros(shape = (num_t, len(celltype)))
    for i in range(ns):
        Prob_single = np.zeros(shape = (num_t, len(celltype)))
        Prob_single[:,np.searchsorted(celltype, cell_type_single[i])] = y_t_pred_softmax[i]
        for j in range(num_t):
            overall_P[j,:] += Prob_single[j,:]
    overall_P = pd.DataFrame(np.array(overall_P))
    for c in range(ct_counts.shape[0]):
        overall_P.iloc[:,c] = overall_P.iloc[:,c]/ct_counts.iloc[c]['num']
            
    # Hard voting for label prediction
    pred_label = np.empty(shape = (num_t, 1), dtype='object')
    for c in range(num_t):
        ct_all = Counter(label_t_pred_mtx.iloc[c,:])
        aa = ct_all.keys()
        bb = ct_all.values()
        adj_prob = np.array(list(bb)) / np.array([ct_counts.loc[ct_counts['ct'] == tt]['num'].values[0] for tt in aa])
        if(len(np.where(adj_prob == max(adj_prob))[0]) == 1):
            pred_label[c] = list(aa)[np.where(adj_prob == max(adj_prob))[0][0]]
        else:
            id = np.where(adj_prob == max(adj_prob))[0]
            score = [np.sum(pd.DataFrame(single_p).iloc[c, np.where(label_t_pred_mtx.iloc[c,:] == tt)[0]]) for tt in np.array(list(aa))[id]]
            pred_label[c] = np.array(list(aa))[id][np.where(score == max(score))[0][0]] 

    # Unseen measurement calculation   
    m = unseen_metric_cal(H_single, overall_P, label_t_pred_mtx, ct_counts)

    if threshold == "default":
        unseen_idx, threshold_df = threshold_selection(m)
    else:
        threshold_df = threshold
        unseen_idx = np.where(m >= np.percentile(sorted(m), (1-threshold_df)*100))[0]

    final_annotation = np.copy(pred_label)
    final_annotation[unseen_idx] = 'unassigned'


    return pred_label, final_annotation, m, threshold_df