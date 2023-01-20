# Ensemble of multiple references for single-cell RNA sequencing data annotation and unseen cell-type identification

mtANN is a novel cell-type annotation framework
that integrates ensemble learning and deep learning
simultaneously, to annotate cells in a new query dataset with the help of multiple well-labeled reference datasets. It takes multiple well-labeled reference datasets and a query dataset that needs to be annotated as input. It begins with generating a series of subsets for each reference dataset by adopting various gene selection methods. Next, for each reference subset, a base classification model is trained based on neural networks. Then, mtANN annotates the cells in the query dataset by integrating the prediction results from all the base classification models. Finally, it identifies cells that may belong to cell types not observed in the reference datasets according to the uncertainty of the predictions.

![Figure1](Figure1.png)

## System Requirements

Python support packages: pandas, numpy, scanpy, scipy, sklearn, torch, giniclust3, rpy2

## Versions the software has been tested on

### Environment 1
- System: Ubuntu 18.04.5
- Python: 3.8.8
- Python packages: pandas = 1.2.3, numpy = 1.19,2, scanpy=1.9.0, scipy = 1.6.1, sklearn = 0.24.1, torch = 1.9.1, giniclust3 = 1.1.0, rpy2 = 3.5.2

# Installation

`pip install -i https://test.pypi.org/simple/ mtANN==0.0.0`

# Useage

`example1` shows data loading and mtANN annotation of query dataset on the Pancreas dataset collection. 

`example2` shows the simulation of unseen cell type and mtANN annotation of query dataset on the PBMC dataset collection. 

 
# Contact

Please do not hesitate to contact Miss Yi-Xuan Xiong ([xyxuana@mails.ccnu.edu.cn](xyxuana@mails.ccnu.edu.cn)) or Dr. Xiao-Fei Zhang ([zhangxf@ccnu.edu.cn](zhangxf@ccnu.edu.cn)) to seek any clarifications regarding any contents or operation of the archive.



