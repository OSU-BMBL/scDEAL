import pandas as pd
import scanpy as sc
import torch


class AnnTorchDataset(torch.utils.data.Dataset):
    """
    pytorch wrapper for AnnData Datasets
    """

    def __init__(self, adata):

        self.adata = adata

    def __getitem__(self, index):
        return self.adata.X[index]

    def __len__(self):
        return self.adata.shape[0]