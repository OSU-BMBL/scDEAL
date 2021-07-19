import scanpy as sc
import preprocessing as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams

def trajetory_paga(adata,clustering = "leiden",n_neighbors = 20):
    
    # PCA 
    sc.tl.pca(adata, svd_solver='arpack')

    # Clustering using leiden method 
    if clustering == "leiden":
        sc.tl.leiden(adata,resolution=0.2)

    # Comstruct neighbour graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    # Initialize run of paga
    sc.tl.paga(adata, groups= clustering)
    sc.pl.paga(adata, threshold=0.03)

    # Use the initialized of paga to draw trajetory
    sc.tl.draw_graph(adata, init_pos='paga')
    sc.pl.draw_graph(adata, color=[clustering], legend_loc='on data')

    return adata
