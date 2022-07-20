# Ensemble of multiple references for single-cell RNA sequencing data annotation and unseen cell-type identification

mtANN is a novel cell-type annotation framework
that integrates ensemble learning and deep learning
simultaneously, to annotate cells in a new query dataset with the help of multiple well-labeled reference datasets. It takes multiple well-labeled reference datasets and a query dataset that needs to be annotated as input. It begins with generating a series of subsets for each reference dataset by adopting various gene selection methods. Next, for each reference subset, a base classification model is trained based on neural networks. Then, mtANN annotates the cells in the query dataset by integrating the prediction results from all the base classification models. Finally, it identifies cells that may belong to cell types not observed in the reference datasets according to the uncertainty of the predictions.

## System Requirements

Python support packages: pandas, numpy, scanpy, scipy, sklearn, torch, giniclust3, rpy2

## Versions the software has been tested on

### Environment 1
- System: Ubuntu 18.04.5
- Python: 3.8.8
- Python packages: pandas = 1.2.3, numpy = 1.19,2, scanpy = 1.7.1, scipy = 1.6.1, sklearn = 0.24.1, torch = 1.9.1, giniclust3 = 1.1.0, rpy2 = 3.5.2

### Environment 2
- System: Mac OS 12.2.1
- Python: 3.7.4
- Python packages: pandas = 1.3.2, numpy = 1.18,1, scanpy = 1.6.0, scipy = 1.4.1, sklearn = 0.22.1, torch = 1.9.0, giniclust3 = 1.1.2, rpy2 = 3.5.2

### Environment 3
- System: Windows 10
- Python: 3.8.3
- Python packages: pandas = 1.3.2, numpy = 1.18,5, scanpy = 1.9.1, scipy = 1.5.0, sklearn = 0.23.1, torch = 1.12.0, giniclust3 = 1.1.2, rpy2 = 3.5.2

# Installation

`pip install mtANN`

# Useage

The `demo.py` file provides a demo of running the mtANN package, and the results can be obtained by running the following code:  
`python demo.py` 
and the results are saved in the results folder, where  
`pred_label.csv` is the annotation results of mtANN,  
`unk_ind.csv` is the index of unseen cells identified by mtANN,  
`entropy.csv` is the measurement of mtANN for identifying unseen cells.
 
Note: For the source data, please download the data and unzip it. Then put them under the same path with `demo.py`.
 
# Contact

Please do not hesitate to contact Miss Yi-Xuan Xiong ([xyxuana@mails.ccnu.edu.cn](xyxuana@mails.ccnu.edu.cn)) or Dr. Xiao-Fei Zhang ([zhangxf@ccnu.edu.cn](zhangxf@ccnu.edu.cn)) to seek any clarifications regarding any contents or operation of the archive.




