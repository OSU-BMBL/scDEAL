import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plot


def read_sc_file(file_path,header=0,index_col=0,sep=None):
    '''
    This is a fucntion to load data having multiple formats.

    Params:
    -------
    
    file_path: str,
        The path of the input file
    
    header: int, (default: `0`)
        Only used if loading txt or csv files. Set the row number of header.
    
    index_col: int, (default: `0`)
        Only used if loading txt or csv files. Set the row name number of the expression file.

    sep: str, (default: `"\t"`)
        Only used if loading txt or csv files. Set the seperator of the input file.

    Return:
    -------
    
    gene_expression: AnnData,
        The load data. 
    '''
    filename = file_path
    separators = ["\t","\n"," ",","] 

    # Read first line to select a right seperator
    def try_seperators(filename, header, index_col, seps):
        for s in seps:
            first_row = pd.read_csv(filename, header=header, index_col=index_col, sep = s, nrows=1)
            if(first_row.shape[1]>0):
                return s
        print("cannot find correct seperators, return tab as seperator")
        return '\t'

    # deal with csv file 
    if ((filename.find(".csv")>=0) or (filename.find(".txt")>=0)):

        # If a seperator is defined
        if(sep!=None):
            counts_drop = pd.read_csv(filename, header=header, index_col=index_col, sep = sep)

        else:
            seperator = try_seperators(filename, header, index_col, separators)
            counts_drop = pd.read_csv(filename, header=header, index_col=index_col, sep = seperator)

        if counts_drop.shape[0]>  counts_drop.shape[1]:
            counts_drop = counts_drop.T
        gene_expression = sc.AnnData(counts_drop)
        
    # deal with txt file
    # elif (filename.find(".txt")>=0):
    #     counts_drop = pd.read_csv(filename, header=header, index_col=index_col, sep= sep)
    #     gene_expression = sc.AnnData(counts_drop.T)
        
    # deal with 10x h5 file
    elif filename.find(".h5")>=0:
        if filename.find(".h5ad")<0:
            gene_expression = sc.read_10x_h5(filename, genome=None, gex_only=True)
        else:
            gene_expression = sc.read_h5ad(filename)
    
    # Deal with 10x mtx files
    else:
        gene_expression = sc.read_10x_mtx(filename,  # the directory with the `.mtx` file 
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True)            

    return gene_expression

def concat(adata_dict,join='inner', sample_key='sample', batch_categories=None, index_unique='-', combat = False, combat_key = "batch",covariates=None):
    '''
    This is a function to concat a dictonary of AnnData objects. Please predefine your own batch keys before the concatination

    Params:
    -------
    adata_dict: `dict`, {"identity":AnnData from scanpy}
        A dictionary: 
        Values should AnnData objects: https://anndata.readthedocs.io/en/latest/api.html#module-anndata
        Keys should be the identitiy of the adata object.
    
    join: `str`, optional (default: `"inner"`)
        Use intersection (``'inner'``) or union (``'outer'``) of variables.
    
    sample_key: `str`, (default: `"sample"`)
        Add the sample annotation to :attr:`obs` using this key.
    
    batch_categories: `str`,optional (default: `None`)
        Use these as categories for the batch annotation. By default, use increasing numbers.
    
    index_unique, `str`, optional (default: `"-"`)
        Make the index unique by joining the existing index names with the
        batch category, using ``index_unique='-'``, for instance. Provide
        ``None`` to keep existing indices.
    
    combat: `bool`, optional (defalut: `False`)
        Decide if to to the batch effect correction
    
    combat_key: `str`, optional (default: `"batch"`)
        Key to a categorical annotation from adata.obs that will be used for batch effect removal

    covariates
        Additional covariates such as adjustment variables or biological condition. Note that
        not including covariates may introduce bias or lead to the removal of biological signal 
        in unbalanced designs.

    inplace: bool, optional (default: `True`)
        Wether to replace adata.X or to return the corrected data

    
    Returns
    -------
    adata: AnnData
        The concatenated AnnData, where "adata.obs[batch_key]"
        stores a categorical variable labeling the original file identity.

    '''
    # Get the list of the dictionary keys to store the original file identity
    ada_keys = list(adata_dict.keys())
    # Concat all the adata in the dictionary 
    adata = adata_dict[ada_keys[0]].concatenate([adata_dict[k] for k in ada_keys[1:]]
                                      ,join=join
                                      ,batch_key=sample_key
                                      ,batch_categories=batch_categories
                                      ,index_unique=index_unique)
    
    if(combat == False):
        return adata
    
    else:
        # Process combat batch correction
        sc.pp.combat(adata, key=combat_key, covariates=covariates)

    return adata

def cal_ncount_ngenes(adata,sparse=False,remove_keys=[]):
    
    mito_genes = (adata.var_names.str.lower().str.rfind('mt-'))!=-1
    rps_genes = (adata.var_names.str.lower().str.rfind('rps'))!=-1
    rpl_genes = (adata.var_names.str.lower().str.rfind('rpl'))!=-1

    adata.var['mt-'] = mito_genes
    adata.var['rps'] = rps_genes
    adata.var['rpl'] = rpl_genes

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt-'], percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['rps'], percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['rpl'], percent_top=None, log1p=False, inplace=True)

    if len(remove_keys)>0:
        mask = np.ones(adata.shape[1])
        if 'mt-' in remove_keys:
            mask = np.logical_and(mask,mito_genes == False)
        if 'rps' in remove_keys:
            mask = np.logical_and(mask,rps_genes == False)
        if 'rpl' in remove_keys:
            mask = np.logical_and(mask,rpl_genes == False)

        adata = adata[:, mask]



    #if sparse == False:    
        #adata.obs['n_counts'] = adata.X.sum(axis=1)
        #adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        #adata.obs['percent_rps'] = np.sum(adata[:, rps_genes].X, axis=1) / np.sum(adata.X, axis=1)
        #adata.obs['percent_rpl'] = np.sum(adata[:, rpl_genes].X, axis=1) / np.sum(adata.X, axis=1)


    #else:
        #adata.obs['n_counts'] = adata.X.sum(axis=1).A1
        #adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
        #adata.obs['percent_rps'] = np.sum(adata[:, rps_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
        #adata.obs['percent_rpl'] = np.sum(adata[:, rpl_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1

    return adata

def receipe_my(adata,l_n_genes = 500, r_n_genes= 5000, filter_mincells=3,filter_mingenes=200, percent_mito = 5, normalize = False,log = False,sparse = False,plotinfo= False,
                remove_genes=[]):

    sc.pp.filter_cells(adata, min_genes=filter_mingenes)
    sc.pp.filter_genes(adata, min_cells=filter_mincells)
    
    adata = cal_ncount_ngenes(adata,remove_keys=remove_genes)

    # if sparse == False:    
    #     adata.obs['n_counts'] = adata.X.sum(axis=1)
    #     adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
    # else:
    #     adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    #     adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1

    adata = adata[
        np.logical_and(
        (adata.obs['n_genes_by_counts'] > l_n_genes), 
        (adata.obs['n_genes_by_counts'] < r_n_genes)),:]
    adata = adata[adata.obs['pct_counts_mt-'] < percent_mito, :]

    if(plotinfo!=False):
        sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt-'],
             jitter=0.4, multi_panel=True, save=True)
        #plt.savefig(plotinfo)



    print(adata.shape)
    
    if normalize == True:
        sc.pp.normalize_total(adata)
    #sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    if log == True:
        sc.pp.log1p(adata)

    #sc.pp.scale(adata, max_value=10)

    return adata
