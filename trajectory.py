import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc
from scipy import stats



def trajectory(adata,now,color="leiden",neigbhor_keys=None,root_key='sensitive',genes_vis=None,root=1,plot=False):
    ##draw
    sc.tl.draw_graph(adata)
    
    if (plot==True):
        sc.pl.draw_graph(adata, color=['leiden','sens_label'], legend_loc='on data',save="Initial_graph_"+now, show=False)

    # Diffusion map graph
    sc.tl.diffmap(adata)
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')
    sc.tl.draw_graph(adata)
    if (plot==True):
        sc.pl.draw_graph(adata, color=['leiden','sens_label'], legend_loc='on data',save="Diffusion_graph_"+now, show=False)

    ## trajectory1
    sc.tl.paga(adata, groups='leiden')
    #if (plot==True):
    sc.pl.paga(adata, color='leiden',save="Paga_initial"+now, show=False) 

    #Recomputing the embedding using PAGA-initialization
    sc.tl.draw_graph(adata, init_pos='paga')
    if (plot==True):
        sc.pl.draw_graph(adata, color=['leiden'], legend_loc='on data',save="Paga_initialization_graph"+now, show=False)
        sc.pl.paga_compare(
            adata,threshold=0.03, title='', right_margin=0.2, size=10,
            edge_width_scale=0.5,legend_fontsize=12, fontsize=12, frameon=False,
            edges=True,save="Paga_cp1"+now, show=False)
    adata.uns['iroot'] = np.flatnonzero(adata.obs[root_key]  == root)[0]
    sc.tl.dpt(adata)
    adata_raw = adata
    sc.pp.log1p(adata_raw)
    sc.pp.scale(adata_raw)
    adata.raw = adata_raw

    if (plot==True):
        sc.pl.draw_graph(adata, color=['sens_preds','1_score','0_score', 'dpt_pseudotime'], legend_loc='on data',save="Pseudotime_graph"+now, show=False)
        sc.pl.paga_compare(
            adata, color="dpt_pseudotime",threshold=0.03, title='', right_margin=0.2, size=10,
            edge_width_scale=0.5,legend_fontsize=12, fontsize=12, frameon=False,
            edges=True,save="Paga_cp2"+now, show=False)
            
        if (genes_vis != None):
            sc.pl.draw_graph(adata, color=genes_vis, legend_loc='on data',save="Pseudotime_graph_genes"+now, show=False)

    
    # Using Trans features
    sc.tl.diffmap(adata,neighbors_key="Trans")
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap',key_added='difftrans')
    sc.tl.draw_graph(adata,neighbors_key='difftrans')
    if (plot==True):
        sc.pl.draw_graph(adata, color=['leiden_trans','sens_label'], legend_loc='on data',save="Diffusion_graph_trans_"+now, show=False)
    ## trajectory1
    sc.tl.paga(adata, groups='leiden_trans',neighbors_key='difftrans')
    if (plot==True):
        sc.pl.paga(adata, color=['leiden_trans'],save="Paga_trans"+now, show=False) 

    #Recomputing the embedding using PAGA-initialization
    sc.tl.draw_graph(adata, init_pos='paga',neighbors_key='difftrans')
    if (plot==True):
        sc.pl.draw_graph(adata, color=['leiden_trans'], legend_loc='on data',save="Paga_trans_graph"+now, show=False)

    corelations={}

    keys = ['sens_preds','rest_preds','Sensitive_score','Resistant_score','1_score','0_score']

    for k in keys:
        try:
            corelations[k] = stats.spearmanr(adata.obs[k],adata.obs['dpt_pseudotime'])
        except:
            print("Key not exist:"+k)
    return adata,corelations
    
