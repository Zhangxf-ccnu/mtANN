import numpy as np
import torch
from collections import Counter
from nipy.algorithms.clustering.ggmixture import GGM


def label_transform_num(label):

    label = np.array(label, dtype="<U128")
    label_set = np.unique(label)
    num_label_set = np.arange(0, len(label_set))

    label_num_transform = dict(zip(label_set, num_label_set))
    num_label_transform = dict(zip(num_label_set, label_set))

    num_label = np.zeros(len(label))

    for s in label_set:
        num_label[np.where(label == s)] = label_num_transform[s]

    return num_label, num_label_transform, label_num_transform



def num_to_label(num_label, num_label_transform):
    
    num_label = np.array(num_label)
    
    label = np.zeros(len(num_label), dtype='<U128')
    
    for s in np.unique(num_label):
        label[np.where(num_label == s)] = num_label_transform[s]
                
    return label


def label_to_num(label, label_num_transform):
    
    label = np.array(label, dtype='<U128')
    
    num_label = np.zeros(len(label))
    
    for s in np.unique(label):
        num_label[np.where(label == s)] = label_num_transform[s]
        
    return num_label




def calculate_weight_s(prediction_softmax_t):
    
    # weight_s = torch.sum(prediction_softmax_t.detach(), dim=0)
    
    label_t_pred = prediction_softmax_t.detach().max(1)[1]
    weight_s = torch.sum(prediction_softmax_t.detach(), dim=0)
    for i in range(len(weight_s)):
        if Counter(label_t_pred)[i] > 0:
            weight_s[i] = weight_s[i] / Counter(label_t_pred)[i]
        else:
            weight_s[i] = weight_s[i] / len(label_t_pred)
    
    weight_s = weight_s / weight_s.max()
    
    return weight_s

def evaluate_t(label_t_u, label_t_pred):
    
    acc_t_with_u = (label_t_u == label_t_pred).mean()
    acc_t_without_u = (label_t_u[np.where(label_t_pred != 'unknown')] == label_t_pred[np.where(label_t_pred != 'unknown')]).mean()
    acc_without_u = (label_t_u[np.where(label_t_u != 'unknown')] == label_t_pred[np.where(label_t_u != 'unknown')]).mean()

    l=np.arange(len(label_t_pred))
    unk_pred = np.where(label_t_pred == "unknown")[0]
    com_pred = np.where(label_t_pred != "unknown")[0]
    unk_true = np.where(label_t_u == "unknown")[0]
    com_true = np.where(label_t_u != "unknown")[0]

    a = np.union1d(unk_pred,unk_true)
    cc = (label_t_pred[np.setdiff1d(l,a)] == label_t_u[np.setdiff1d(l,a)]).sum()/len(label_t_pred)
    misc = (label_t_pred[np.setdiff1d(l,a)] != label_t_u[np.setdiff1d(l,a)]).sum()/len(label_t_pred)
    ia = len(np.intersect1d(unk_true, com_pred))/len(label_t_pred)
    iu = len(np.intersect1d(com_true, unk_pred))/len(label_t_pred)
    cu = len(np.intersect1d(unk_true, unk_pred))/len(label_t_pred)

    tp = len(np.intersect1d(unk_pred, unk_true))
    fp = len(np.intersect1d(unk_pred, com_true)) 
    fn = len(np.intersect1d(com_pred, unk_true))
    tn = len(np.intersect1d(com_pred, com_true))
    pre = tp/(tp + fp + 1e-10)
    rec = tp/(tp + fn + 1e-10)
    f1 = 2 * pre * rec / (pre + rec + 1e-10)

    return acc_t_with_u, acc_t_without_u, acc_without_u, cc, misc, iu, ia, cu, f1

def gaussian_gamma_mixture(x):
    
    ggm = GGM()
    loglik = ggm.estimate(x)
    posterior = ggm.posterior(x)
    bic = 5 * np.log(len(x)) - 2 * loglik * len(x)
    # bic = 5 * 2 - 2 * loglik * len(x)
    
    return posterior[0], bic, ggm, loglik * len(x)

def redu2(lll):
    A = {}
    for i in range(len(lll)):
        for k,v in lll[i].items():
            A[k] = A.get(k,0)+v

    return A