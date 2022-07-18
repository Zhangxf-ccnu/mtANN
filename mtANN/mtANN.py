import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import pandas as pd
import os
import math
from collections import Counter
import random
# from models import NetEncoder, NetDecoder, NetClassifier
# from train import train_ae, train_s
# from utils import label_transform_num, num_to_label
# import params
from mtANN import *
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture


def mtANN(expression_s, label_s, expression_t, t_cell_names):
    
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if params.CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    ns = len(expression_s)
    # label_s_array = []
    # for i in range(ns):
    #     label_s_array.append(np.array(label_s[i]))
    
    # label_t = np.array(label_t, dtype='<U128')
    celltype = list()
    for i in range(len(label_s)):
        celltype.extend(np.unique(label_s[i]))
    celltype = np.array(sorted(np.unique(celltype)))
    
    ct_in_data_counts = []
    for i in range(len(celltype)):
        a=[]
        for j in range(len(label_s)):
            a.append((label_s[j] == celltype[i]).any() * 1)
        ct_in_data_counts.append(np.sum(np.array(a)))
    ct_counts = pd.DataFrame(zip(celltype, ct_in_data_counts))
    ct_counts.columns = ["ct","num"]
    
    
    classifier_s = {}
    encoder = {}
    label_t_pred = {}
    y_t_pred = {}
    y_t_pred_softmax = {}
    H_single = {}
    cell_type_single = {}
    
    l_t_pred = pd.DataFrame(columns=(range(ns)), index=t_cell_names)
    single_p = np.zeros(shape = (expression_t[0].shape[0], ns))
    for i in range(ns):
        print("i={}".format(i))
        num_label_s, num_label_transform, label_num_transform = label_transform_num(label_s[i])
        num_label = len(set(label_s[i]))
        num_gene = expression_s[i].shape[1]

        batch_size_s = int(math.ceil(max(Counter(label_s[i]).values())/10))
        batch_size_t = int(np.min([math.ceil(expression_t[i].shape[0]/10),600]))
        
        if issparse(expression_s[i]):
            expression_s[i] = expression_s[i].tocoo()
            row_s = torch.from_numpy(expression_s[i].row).long()
            col_s = torch.from_numpy(expression_s[i].col).long()
            val_s = torch.from_numpy(expression_s[i].data).float()
            shape_s = expression_s[i].shape
            x_s = torch.sparse_coo_tensor(torch.vstack((row_s, col_s)), val_s, shape_s)
        else:
            x_s = torch.from_numpy(expression_s[i]).float()

        if issparse(expression_t[i]):
            expression_t[i] = expression_t[i].tocoo()
            row_t = torch.from_numpy(expression_t[i].row).long()
            col_t = torch.from_numpy(expression_t[i].col).long()
            val_t = torch.from_numpy(expression_t[i].data).float()
            shape_t = expression_t[i].shape
            x_t = torch.sparse_coo_tensor(torch.vstack((row_t, col_t)), val_t, shape_t)
        else:
            x_t = torch.from_numpy(expression_t[i]).float()

        y_s = torch.from_numpy(num_label_s).long()

        if params.CUDA:
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

        encoder[i] = NetEncoder(num_gene, params.K[0], params.K[1])
        decoder = NetDecoder(params.K[1], params.K[0], num_gene)
        classifier_s[i] = NetClassifier(params.K[1], num_label)

        if params.CUDA:
            encoder[i] = encoder[i].cuda()
            decoder = decoder.cuda()
            classifier_s[i] = classifier_s[i].cuda()
        
        encoder[i], decoder = train_ae(encoder[i], decoder, dataloader_s, dataloader_t)
        encoder[i], decoder, classifier_s[i] = train_s(encoder[i], decoder, classifier_s[i], dataloader_s, dataloader_t)

        encoder[i].eval()
        classifier_s[i].eval()
        with torch.no_grad():
            y_s_pred = classifier_s[i](encoder[i](x_s)).detach().max(1)[1]
            acc_s = (y_s_pred == y_s).float().mean().item()
            print("Final source acc =%.5f" % acc_s)
        
            y_t_pred_softmax[i] = F.softmax(classifier_s[i](encoder[i](x_t)), dim=1).detach().cpu().numpy()
            y_t_pred[i] = classifier_s[i](encoder[i](x_t)).detach().max(1)[1].cpu().numpy()
            label_t_pred[i] = num_to_label(y_t_pred[i], num_label_transform)

            single_p[:,i] = np.max(y_t_pred_softmax[i], axis = 1)

            a=[]
            for t in range(len(t_cell_names)):
                if t_cell_names[t] in l_t_pred.index:
                    a.append(list(l_t_pred.index).index(t_cell_names[t]))
            l_t_pred.iloc[np.array(a),i] = label_t_pred[i]

            h = np.zeros(y_t_pred_softmax[i].shape[0])
            for c in range(y_t_pred_softmax[i].shape[0]):
                a = y_t_pred_softmax[i][c,:][np.where(y_t_pred_softmax[i][c,:] != 0)[0]]
                h[c] = (-a*np.log2(a)).sum()/np.log2(len(a))
            H_single[i] = h
            cell_type_single[i] = num_to_label(np.unique(num_label_s), num_label_transform)        

    P = np.zeros(shape = (len(H_single[0]), len(celltype)))
    for i in range(ns):
        P_overall = np.zeros(shape = (len(H_single[i]), len(celltype)))
        P_overall[:,np.searchsorted(celltype, cell_type_single[i])] = y_t_pred_softmax[i]
        for j in range(len(H_single[i])):
            P[j,:] += P_overall[j,:]
            
    ####################hard voting for label prediction####################
    pred_label = np.empty(shape = (pd.DataFrame(l_t_pred).shape[0], 1), dtype='object')
    for c in range(pd.DataFrame(l_t_pred).shape[0]):
        ct_all = Counter(pd.DataFrame(l_t_pred).iloc[c,:])
        aa = ct_all.keys()
        bb = ct_all.values()
        adj_prob = np.array(list(bb)) / np.array([ct_counts.loc[ct_counts['ct'] == tt]['num'].values[0] for tt in aa])
        if(len(np.where(adj_prob == max(adj_prob))[0]) == 1):
            pred_label[c] = list(aa)[np.where(adj_prob == max(adj_prob))[0][0]]
        else:
            id = np.where(adj_prob == max(adj_prob))[0]
            score = [np.sum(pd.DataFrame(np.array(single_p)).iloc[c, np.where(pd.DataFrame(l_t_pred).iloc[c,:] == tt)[0]]) for tt in np.array(list(aa))[id]]
            pred_label[c] = list(aa)[np.where(score == max(score))[0][0]] 
                
    en = pd.DataFrame(np.array(list(H_single.values())))
    en.iloc[np.where(np.isnan(en))] = 0
    ###################1. calculate entropy with singleh####################
    entropy = np.mean(np.array(en), axis = 0)
    entropy1 = entropy.reshape(len(entropy), 1)
    entropy1 = np.array(entropy1).reshape(len(entropy1), 1)

    overall_P = pd.DataFrame(np.array(P))
    ####################2. calculate entropy with overall prob####################
    for c in range(ct_counts.shape[0]):
        overall_P.iloc[:,c] = overall_P.iloc[:,c]/ct_counts.iloc[c]['num']
        
    entropy2 = []
    for l in range(overall_P.shape[0]):
        a = overall_P.iloc[l,np.where(overall_P.iloc[l,:] != 0)[0]].values
        a = a/np.sum(a)
        entropy2.append((-a*np.log2(a)).sum()/np.log2(overall_P.shape[1]))
    entropy2=np.array(entropy2)
    entropy2 = np.array(entropy2).reshape(len(entropy2), 1)
        
    ####################3. calculate entropy with label prediction####################
    entropy3 = []
    for j in range(len(pred_label)):
        keys = []
        values = []
        for k, v in Counter(l_t_pred.iloc[j,:]).items():
            keys.append(k)
            values.append(v)
        count = np.array(values)
        ct = np.array(keys)
        count = count / np.array([ct_counts.loc[ct_counts['ct'] == tt]['num'] for tt in ct])
        p_count = count/sum(count)
        entropy3.append((-p_count*np.log2(p_count)).sum()/np.log2(ct_counts.shape[0]))
    entropy3 = np.array(entropy3).reshape(len(entropy3), 1)
    
    entropy1 = (entropy1 - np.min(entropy1))/(np.max(entropy1) - np.min(entropy1))
    entropy2 = (entropy2 - np.min(entropy2))/(np.max(entropy2) - np.min(entropy2))
    entropy3 = (entropy3 - np.min(entropy3))/(np.max(entropy3) - np.min(entropy3))
    entropy = (entropy1+entropy2+entropy3)/3
    
    d = [1,2,3,4,5]
    aic = []
    for dd in d:
        gm = GaussianMixture(n_components=dd, random_state=0).fit(entropy)
        aic.append(gm.aic(entropy))
    
    if (aic.index(np.min(aic))+1 == 1):
        ind_pred_df = []
    else:
        gm = GaussianMixture(n_components=(aic.index(np.min(aic))+1), random_state=0).fit(entropy)
        group = gm.predict(entropy)
        score = [np.mean(entropy[np.where(group == g)]) for g in np.unique(group)]
        if (len(np.where(np.array(score) >= 0.6)[0]) <= 1):
            ind_pred_df = np.where(group == score.index(np.max(score)))[0]
        else:
            gg = np.where(np.array(score) >= 0.6)[0]
            ind_pred_df = np.concatenate(([np.where(group == i)[0] for i in gg]))
    
    return pred_label, ind_pred_df, entropy