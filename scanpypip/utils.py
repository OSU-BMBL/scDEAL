import pandas as pd
import scanpy as sc
import torch
def get_de_dataframe(adata,index):
        df_result = pd.DataFrame({
            "names":[item[index] for item in adata.uns["rank_genes_groups"]["names"]]
            ,"score":[item[index] for item in adata.uns["rank_genes_groups"]["scores"]]
            ,"logfoldchanges":[item[index] for item in adata.uns["rank_genes_groups"]["logfoldchanges"]]
            , "pvals":[item[index] for item in adata.uns["rank_genes_groups"]["pvals"]]
            , "pvals_adj":[item[index] for item in adata.uns["rank_genes_groups"]["pvals_adj"]]
             })
        
        return df_result