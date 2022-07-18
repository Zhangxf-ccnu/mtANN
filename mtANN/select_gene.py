import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import scanpy as sc
import numpy as np
import giniclust3 as gc
import anndata


def select_gene(exprsMat, trainClass, feature):

    if feature in ["de", "dv", "dd", "dp", "bi"]:

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_exprsMat = ro.conversion.py2rpy(exprsMat.T)

        r_trainClass = ro.StrVector(trainClass)
        r_feature = ro.StrVector([feature])

        ro.r.source("featureSelection.R")
        r_result = ro.r["featureSelection"](r_exprsMat, r_trainClass, r_feature)
        result = list(r_result)
    

    elif feature == "gc":

        adataRaw = sc.AnnData(exprsMat)
        raw_genes = adataRaw.var.index
        sc.pp.normalize_per_cell(adataRaw, counts_per_cell_after=1e4)
        gc.gini.calGini(adataRaw, selection='p_value', p_value=0.01, min_gini_value=0.5)
        res_gene = adataRaw.var['gini']
        gc_result = raw_genes[np.where(res_gene == True)[0]]
        result = list(gc_result)


    elif feature == "disp":

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_exprsMat = ro.conversion.py2rpy(exprsMat.T)

        r_trainClass = ro.StrVector(trainClass)

        r_script = '''
            library(Seurat)
            disp_seurat <- function(data, labels){
                seurat_object <- CreateSeuratObject(data)
                names(labels) <- colnames(seurat_object)
                seurat_object <- AddMetaData(seurat_object, as.character(labels), 'celltype')
                seurat_object <- NormalizeData(seurat_object, verbose = FALSE)
                seurat_object = FindVariableFeatures(seurat_object, selection.method = "disp", nfeatures = 500)
                genes = seurat_object@assays$RNA@var.features
                return(genes)
            }
        '''

        ro.r(r_script)
        disp_result = ro.r['disp_seurat'](r_exprsMat, r_trainClass)
        result = list(disp_result)


    elif feature == "vst":

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_exprsMat = ro.conversion.py2rpy(exprsMat.T)

        r_trainClass = ro.StrVector(trainClass)

        r_script = '''
            library(Seurat)
            vst_seurat <- function(data, labels){
                seurat_object <- CreateSeuratObject(data)
                names(labels) <- colnames(seurat_object)
                seurat_object <- AddMetaData(seurat_object, as.character(labels), 'celltype')
                seurat_object <- NormalizeData(seurat_object, verbose = FALSE)
                seurat_object = ScaleData(seurat_object)
                seurat_object = FindVariableFeatures(seurat_object, nfeatures = 500)
                genes = seurat_object@assays$RNA@var.features
                return(genes)
            }
        '''

        ro.r(r_script)
        vst_result = ro.r['vst_seurat'](r_exprsMat, r_trainClass)
        result = list(vst_result)


    return(result)




def select_gene_all(exprsMat, trainClass):

    gene_dict = {}

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_exprsMat = ro.conversion.py2rpy(exprsMat.T)

    r_trainClass = ro.StrVector(trainClass)


    ro.r.source("featureSelection.R")

    for feature in ["limma", "DV", "DD", "chisq", "BI"]:
        print("gene={}".format(feature))
        r_feature = ro.StrVector([feature])
        r_result = ro.r["featureSelection"](r_exprsMat, r_trainClass, r_feature)
        gene_dict[feature] = list(r_result)


    r_script_disp = '''
        library(Seurat)
        disp_seurat <- function(data, labels){
            seurat_object <- CreateSeuratObject(data)
            names(labels) <- colnames(seurat_object)
            seurat_object <- AddMetaData(seurat_object, as.character(labels), 'celltype')
            seurat_object@assays$RNA@data = log1p(seurat_object@assays$RNA@counts)
            seurat_object = FindVariableFeatures(seurat_object, selection.method = "disp", nfeatures = 500)
            genes = seurat_object@assays$RNA@var.features
            return(genes)
        }
    '''

    ro.r(r_script_disp)
    disp_result = ro.r["disp_seurat"](r_exprsMat, r_trainClass)
    gene_dict["disp"] = list(disp_result)


    r_script_vst = '''
        library(Seurat)
        vst_seurat <- function(data, labels){
            seurat_object <- CreateSeuratObject(data)
            names(labels) <- colnames(seurat_object)
            seurat_object <- AddMetaData(seurat_object, as.character(labels), 'celltype')
            seurat_object = ScaleData(seurat_object)
            seurat_object = FindVariableFeatures(seurat_object, nfeatures = 500)
            genes = seurat_object@assays$RNA@var.features
            return(genes)
        }
    '''

    ro.r(r_script_vst)
    vst_result = ro.r["vst_seurat"](r_exprsMat, r_trainClass)
    gene_dict["vst"] = list(vst_result)

    
    adataRaw = sc.AnnData(exprsMat)
    raw_genes = adataRaw.var.index
    sc.pp.normalize_per_cell(adataRaw, counts_per_cell_after=1e4)
    gc.gini.calGini(adataRaw, selection='p_value', p_value=0.01, min_gini_value=0.5)
    res_gene = adataRaw.var['gini']
    gc_result = raw_genes[np.where(res_gene == True)[0]]
    gene_dict["gc"] = list(gc_result)
    
    
    return(gene_dict)







# from preprocess import preprocess, readfile_csv, read_label
# source_file = {"exp_filename": "/Users/xyxuan/Downloads/mtANN/datasets/panc/{}.csv".format("xin"), "type_filename": "/Users/xyxuan/Downloads/mtANN/datasets/panc/{}_label.csv".format("xin")}
# source_dataset = readfile_csv(source_file)

# res = select_gene_all(source_dataset["expression"], source_dataset["cell_type"])

# g1 = pd.read_csv("/Users/xyxuan/Downloads/mtANN/datasets/panc/gene/_xin_disp_.csv", header=0, index_col=0)
# sum(sum(np.array(res['disp']) == np.array(g1)))
# g1.shape