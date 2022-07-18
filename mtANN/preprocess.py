import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, minmax_scale
import qnorm


def readfile_csv(Filename):
    
    dataset = {}
       
    dataset["expression"] = pd.read_csv(Filename["exp_filename"], header=0, index_col=0)
    dataset["cell_type"] = pd.read_csv(Filename["type_filename"], header=0, index_col=False).x.values

    # dataset["expression"] = np.array(dataset["expression"])
    dataset["expression"] = dataset["expression"].div(dataset["expression"].apply(lambda x: x.sum(), axis=1), axis=0) * 10000
    # dataset["expression"] = pd.DataFrame(dataset["expression"])

    return dataset

def read_label(Filename):
       
    label = pd.read_csv(Filename["type_filename"], header=0, index_col=False).x.values
    
    return label



def readfile(source_file, target_file):
     
    source_dataset = readfile_csv(source_file["exp_filename"], source_file["type_filename"])
    target_dataset = readfile_csv(target_file["exp_filename"], target_file["type_filename"])
        
    return source_dataset, target_dataset



def closed(source_dataset, target_dataset):
    
    source_celltype = np.unique(source_dataset["cell_type"])
    target_celltype = np.unique(target_dataset["cell_type"])
    celltype = set(source_celltype).intersection(target_celltype)

    use_index_source = np.sort(np.concatenate([np.where(source_dataset["cell_type"] == i) for i in celltype], axis=1)[0])
    source_dataset["expression"] = source_dataset["expression"].iloc[use_index_source, :]
    source_dataset["cell_type"] = source_dataset["cell_type"][use_index_source]

    use_index_target = np.sort(np.concatenate([np.where(target_dataset["cell_type"] == i) for i in celltype], axis=1)[0])
    target_dataset["expression"] = target_dataset["expression"].iloc[use_index_target, :]
    target_dataset["cell_type"] = target_dataset["cell_type"][use_index_target]
    
    return source_dataset, target_dataset



# Get common genes, normalize  and scale the sets
def scale_sets(sets):
    # input -- a list of all the sets to be scaled
    # output -- scaled sets
    common_genes = set(sets[0].index)
    for i in range(1, len(sets)):
        common_genes = set.intersection(set(sets[i].index),common_genes)
    common_genes = sorted(list(common_genes))
    sep_point = [0]
    for i in range(len(sets)):
        sets[i] = sets[i].loc[common_genes,]
        sep_point.append(sets[i].shape[1])
    total_set = np.array(pd.concat(sets, axis=1, sort=False), dtype=np.float32)
    total_set = np.divide(total_set, np.sum(total_set, axis=0, keepdims=True)) * 10000
    total_set = np.nan_to_num(total_set, nan=0)
    total_set = np.log2(total_set+1)
    expr = np.sum(total_set, axis=1)
    total_set = total_set[np.logical_and(expr > np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
    cv = np.std(total_set, axis=1) / (np.mean(total_set, axis=1)+1e-10)
    total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
    for i in range(len(sets)):
        sets[i] = total_set[:, sum(sep_point[:(i+1)]):sum(sep_point[:(i+2)])]
    return sets



def preprocessing(source_expression, target_dataset):
    
    ns = len(source_expression)
    sets = []
    for i in range(ns):
        sets.append(source_expression[i].T)
        
    sets.append(target_dataset["expression"].T)
    sets = scale_sets(sets)
    
    source_expression = []
    target_dataset_ = {}
    for i in range(ns):
        source_expression.append(sets[i].T)

    target_dataset_['expression'] = sets[-1].T
    target_dataset_['cell_type'] = target_dataset['cell_type']
    
    return source_expression, target_dataset_

def preprocess(source_dataset, target_dataset, take_log = True, standardization=True, scaling=True, q_normlize = True):

    # source_dataset["expression"] = np.array(source_dataset["expression"])
    # target_dataset["expression"] = np.array(target_dataset["expression"])

    if take_log:
        source_dataset["expression"] = (source_dataset["expression"]+1).apply(np.log2)
        target_dataset["expression"] = (target_dataset["expression"]+1).apply(np.log2)
    if standardization:
        source_dataset["expression"] = source_dataset["expression"].apply(scale, axis=0)
        target_dataset["expression"] = target_dataset["expression"].apply(scale, axis=0)
    if scaling:
        source_dataset["expression"] = minmax_scale(source_dataset["expression"], feature_range=(0, 1), axis=0)
        target_dataset["expression"] = minmax_scale(target_dataset["expression"], feature_range=(0, 1), axis=0)
    if q_normlize:
        source_dataset["expression"] = qnorm.quantile_normalize(source_dataset["expression"], axis=1)
        target_dataset["expression"] = qnorm.quantile_normalize(target_dataset["expression"], axis=1)
    
    source_dataset["expression"] = np.array(source_dataset["expression"])
    target_dataset["expression"] = np.array(target_dataset["expression"])
        
    return source_dataset, target_dataset

def preprocess_covid(source_dataset, take_log = True, standardization=True, scaling=True, q_normlize = True):

    if take_log:
        source_dataset["expression"] = (source_dataset["expression"]+1).apply(np.log2)
    if standardization:
        source_dataset["expression"] = source_dataset["expression"].apply(scale, axis=0)
    if scaling:
        source_dataset["expression"] = minmax_scale(source_dataset["expression"], feature_range=(0, 1), axis=0)
    if q_normlize:
        source_dataset["expression"] = qnorm.quantile_normalize(source_dataset["expression"], axis=1)
    
    source_dataset["expression"] = np.array(source_dataset["expression"])
        
    return source_dataset
    



# if __name__ == '__main__':
#     import os
#     os.chdir("E:\\Machine Learning\\paper\\WMG_2\\code_test\\test\\test_preprocess")
    
#     source_file = {"exp_filename": "panc\\muraro.csv", "type_filename": "panc\\muraro_label.csv"}
#     target_file = {"exp_filename": "panc\\baron_human.csv", "type_filename": "panc\\baron_human_label.csv"}
    
#     source_dataset, target_dataset = readfile(source_file, target_file)
#     source_dataset, target_dataset = closed(source_dataset, target_dataset)
#     source_dataset, target_dataset = preprocessing(source_dataset, target_dataset)
    
#     source_expression = source_dataset["expression"]
#     source_label = source_dataset["cell_type"]
#     target_expression = target_dataset["expression"]
#     target_label = target_dataset["cell_type"]
    
    
    
    
    
    
    
    
    
    
    
    
