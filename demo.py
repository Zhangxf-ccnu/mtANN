
import os
import pandas as pd
import numpy as np
from mtANN.mtANN import mtANN
from mtANN.preprocess import preprocess, readfile_csv
from mtANN.select_gene import select_gene
import scanpy as sc
from collections import Counter

D = []
for file in os.listdir("./datasets/"):
    if len(file.split("_")) == 2:
        D.append(file.split("_")[0])

target = D[0]
source = list(np.copy(D))
source.remove(target)

features = ["de", "dv", "dd", "dp", "bi","gc","disp","vst"]


epr_s = list()
epr_t = list()
label_s = list()
t_cell_names = list()
res = list()
for ds in range(len(source)):
    for g in range(len(features)):
        print("target={}, source={}, gene={}".format(target,source[ds],features[g]))
        
        source_file = {"exp_filename": "./datasets/{}.gz".format(source[ds]), "type_filename": "./datasets/{}_label.gz".format(source[ds])}
        source_dataset = readfile_csv(source_file)
        target_file = {"exp_filename": "./datasets/{}.gz".format(target), "type_filename": "./datasets/{}_label.gz".format(target)}
        target_dataset = readfile_csv(target_file)
        
        t_cell_names = np.copy(target_dataset["expression"].index)
        
        # res = select_gene(source_dataset['expression'], source_dataset['cell_type'], feature = features[g])
        res.append(pd.read_csv("./datasets/genes/_{}_{}_.csv".format(source[ds], features[g]), header=0, index_col=0)["x"])
        gene_select = set(res[ds*len(features) + g]).intersection(set(target_dataset["expression"].columns))
        
        source_dataset["expression"] = source_dataset["expression"][np.array(list(gene_select))]
        target_dataset["expression"] = target_dataset["expression"][np.array(list(gene_select))]
        
        
        data_s, data_t = preprocess(source_dataset, target_dataset, scaling = True, q_normlize=False)
        epr_s.append(data_s["expression"])
        epr_t.append(data_t["expression"])
        label_s.append(data_s["cell_type"])
                        
pred_label, ind_pred_df, entropy = mtANN(expression_s = epr_s, label_s=label_s, expression_t=epr_t, t_cell_names=t_cell_names)

if os.path.exists('results'):
    pass
else:
    os.mkdir("results")

pd.DataFrame(pred_label).to_csv("results/pred_label.csv", index=False)
pd.DataFrame(np.array(ind_pred_df)).to_csv("results/unk_ind.csv", index=False)
pd.DataFrame(entropy).to_csv("results/entropy.csv", index=True)
