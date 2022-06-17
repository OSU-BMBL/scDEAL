#!/usr/bin/env python
# coding: utf-8

# In[123]:


#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import sys
import time
from decimal import Decimal
import glob

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from captum.attr import IntegratedGradients
from numpy.lib.function_base import gradient
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import (auc, average_precision_score,
                             classification_report, mean_squared_error,
                             precision_recall_curve, r2_score, roc_auc_score)
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import DaNN.mmd as mmd
import scanpypip.preprocessing as pp
import trainers as t
import utils as ut
from models import (AEBase, CVAEBase, DaNN, Predictor, PretrainedPredictor,
                    PretrainedVAEPredictor, TargetModel, VAEBase)
from scanpypip.utils import get_de_dataframe
from trajectory import trajectory
from sklearn.feature_selection import SelectKBest,SelectFdr
from sklearn.feature_selection import chi2


DATA_MAP={
"GSE117872":"data/GSE117872/GSE117872_good_Data_TPM.txt",
"GSE117309":'data/GSE117309/filtered_gene_bc_matrices_HBCx-22/hg19/',
"GSE117309_TAMR":'data/GSE117309/filtered_gene_bc_matrices_HBCx22-TAMR/hg19/',
"GSE121107":'data/GSE121107/GSM3426289_untreated_out_gene_exon_tagged.dge.txt',
"GSE121107_1H":'data/GSE121107/GSM3426290_entinostat_1hr_out_gene_exon_tagged.dge.txt',
"GSE121107_6H":'data/GSE121107/GSM3426291_entinostat_6hr_out_gene_exon_tagged.dge.txt',
"GSE111014":'data/GSE111014/',
"GSE110894":"data/GSE110894/GSE110894.csv",
"GSE122843":"data/GSE122843/GSE122843.txt",
"GSE112274":"data/GSE112274/GSE112274_cell_gene_FPKM.csv",
"GSE116237":"data/GSE116237/GSE116237_bulkRNAseq_expressionMatrix.txt",
"GSE108383":"data/GSE108383/GSE108383_Melanoma_fluidigm.txt",
"GSE140440":"data/GSE140440/GSE140440.csv",
"GSE129730":"data/GSE129730/GSE129730.h5ad",
"GSE149383":"data/GSE149383/erl_total_data_2K.csv",
"GSE110894_small":"data/GSE110894/GSE110894_small.h5ad",
"MIX-Seq":"data/10298696"

}


# In[124]:


files = glob.glob("saved/models/1214*")


# In[125]:


files


# In[126]:


SELECTED_FILES = 5


# # Load arguments

# In[127]:


parser = argparse.ArgumentParser()
# data 
parser.add_argument('--source_data', type=str, default='data/ALL_expression.csv')
parser.add_argument('--label_path', type=str, default='data/ALL_label_binary_wf.csv')
parser.add_argument('--target_data', type=str, default="GSE110894")
parser.add_argument('--drug', type=str, default='I-BET-762')
parser.add_argument('--missing_value', type=int, default=1)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--valid_size', type=float, default=0.2)
parser.add_argument('--var_genes_disp', type=float, default=0)
parser.add_argument('--min_n_genes', type=int, default=0)
parser.add_argument('--max_n_genes', type=int, default=20000)
parser.add_argument('--min_g', type=int, default=200)
parser.add_argument('--min_c', type=int, default=3)
parser.add_argument('--cluster_res', type=float, default=0.3)
parser.add_argument('--remove_genes', type=int, default=1)
parser.add_argument('--mmd_weight', type=float, default=0.25)

# train
parser.add_argument('--source_model_path','-s', type=str, default='saved/models/BET_dw_256_AE.pkl')
parser.add_argument('--target_model_path', '-p',  type=str, default='saved/models/GSE110894_I-BET-762256AE')
parser.add_argument('--pretrain', type=str, default='saved/models/GSE110894_I-BET-762_256_ae.pkl')
parser.add_argument('--transfer', type=str, default="DaNN")

parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--bottleneck', type=int, default=256)
parser.add_argument('--dimreduce', type=str, default="DAE")
parser.add_argument('--predictor', type=str, default="DNN")
parser.add_argument('--freeze_pretrain', type=int, default=0)
parser.add_argument('--source_h_dims', type=str, default="256,256")
parser.add_argument('--target_h_dims', type=str, default="256,256")
parser.add_argument('--p_h_dims', type=str, default="128,64")
parser.add_argument('--predition', type=str, default="classification")
parser.add_argument('--VAErepram', type=int, default=1)
parser.add_argument('--batch_id', type=str, default="HN137")
parser.add_argument('--load_target_model', type=int, default=0)
parser.add_argument('--GAMMA_mmd', type=int, default=1000)
parser.add_argument('--dropout', type=float, default=0.3,help='dropout')

parser.add_argument('--runs', type=int, default=1)

# Analysis
parser.add_argument('--n_DL_genes', type=int, default=50)
parser.add_argument('--n_DE_genes', type=int, default=50)


# misc
parser.add_argument('--message', '-m',  type=str, default='message')
parser.add_argument('--output_name', '-n',  type=str, default='saved/results')
parser.add_argument('--logging_file', '-l',  type=str, default='saved/logs/transfer_')

#
args, unknown = parser.parse_known_args()


# In[128]:


## Testing the args covering
selected_model = files[SELECTED_FILES]
split_name = selected_model.split("_")
print(split_name[1::2])
print(split_name[0::2])


# In[129]:


paras = (split_name[1::2])
para_names = (split_name[0::2])

args.source_h_dims = paras[3]
args.target_h_dims = paras[3]
args.p_h_dims = paras[4]
args.bottleneck = int(paras[2])
args.drug = paras[1]
args.dropout = float(paras[6])

if(paras[0].find("GSE117872")>=0):
    args.target_data = "GSE117872"
    args.batch_id = paras[0].split("GSE117872")[1]
elif(paras[0].find("MIX-Seq")>=0):
    args.target_data = "MIX-Seq"
    args.batch_id = paras[0].split("MIX-Seq")[1]    
else:
    args.target_data = paras[0]
    
args.target_model_path = selected_model


# In[130]:


paras


# In[131]:


# Read parameters
data_name = args.target_data
epochs = args.epochs
dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
na = args.missing_value
data_path = DATA_MAP[args.target_data]
test_size = args.test_size
select_drug = args.drug
freeze = args.freeze_pretrain
valid_size = args.valid_size
g_disperson = args.var_genes_disp
min_n_genes = args.min_n_genes
max_n_genes = args.max_n_genes
source_model_path = args.source_model_path
target_model_path = args.target_model_path 
log_path = args.logging_file
batch_size = args.batch_size
encoder_hdims = args.source_h_dims.split(",")
encoder_hdims = list(map(int, encoder_hdims))
source_data_path = args.source_data 
pretrain = args.pretrain
prediction = args.predition
data_name = args.target_data
label_path = args.label_path
reduce_model = args.dimreduce
predict_hdims = args.p_h_dims.split(",")
predict_hdims = list(map(int, predict_hdims))
leiden_res = args.cluster_res
load_model = bool(args.load_target_model)


# In[132]:


if(args.target_data!="GSE117872"):
    adata_path = glob.glob("D://pyws//trainsource//saved//adata//review//*"+args.target_data+"*")
else:
    adata_path = glob.glob("D://pyws//trainsource//saved//adata//review//*"+args.batch_id+"*")


# In[133]:


adata_final = sc.read_h5ad(adata_path[0])


# In[134]:


if(data_name!="MIX-Seq"):
    adata = pp.read_sc_file(data_path)

    if data_name == 'GSE117872':
        adata =  ut.specific_process(adata,dataname=data_name,select_origin=args.batch_id)
    elif data_name =='GSE122843':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE110894':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE112274':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE116237':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE108383':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE140440':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE129730':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE149383':
        adata =  ut.specific_process(adata,dataname=data_name)
    else:
        adata=adata
else:
    # Can be expt1, expt3, and expt10
    # Add process mix-seq
    expID = args.batch_id
    drug_path = args.drug.capitalize()
    adata = ut.process_mix_seq(drug=drug_path,expt=expID)


# In[135]:


#adata_final.var.index.intersection(adata.var.index)


# In[136]:


adata = adata[adata_final.obs.index,adata_final.var.index]


# In[137]:


adata


# In[138]:


adata_final


# In[139]:


if(data_name!="GSE149383"):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

adata = pp.cal_ncount_ngenes(adata)
adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=0,percent_mito = 100,
                        filter_mingenes=0,normalize=True,log=True)


# In[140]:


if(data_name!="GSE149383"):

    sc.pp.highly_variable_genes(adata,min_disp=g_disperson,max_disp=np.inf,max_mean=6)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
else:
    adata.raw = adata

data=adata.X


# In[141]:


adata


# In[142]:


sc.tl.pca(adata,svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10)
# Generate cluster labels
sc.tl.leiden(adata,resolution=leiden_res)
sc.tl.umap(adata)
adata.obs['leiden_origin']= adata.obs['leiden']
adata.obsm['X_umap_origin']= adata.obsm['X_umap']
data_c = adata.obs['leiden'].astype("long").to_list()


# In[143]:


mmscaler = preprocessing.MinMaxScaler()

try:
    data = mmscaler.fit_transform(data)

except:
    logging.warning("Sparse data , transfrom to dense")

    # Process sparse data
    data = data.todense()
    data = mmscaler.fit_transform(data)


# In[144]:


select_drug


# In[145]:


Xtarget_train, Xtarget_valid, Ctarget_train, Ctarget_valid = train_test_split(data,data_c, test_size=valid_size, random_state=42)


# Select the device of gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# Assuming that we are on a CUDA machine, this should print a CUDA device:
#logging.info(device)
print(device)
torch.cuda.set_device(device)

# Construct datasets and data loaders
Xtarget_trainTensor = torch.FloatTensor(Xtarget_train).to(device)
Xtarget_validTensor = torch.FloatTensor(Xtarget_valid).to(device)
#print(Xtarget_validTensor.shape)
# Use leiden label if CVAE is applied 
Ctarget_trainTensor = torch.LongTensor(Ctarget_train).to(device)
Ctarget_validTensor = torch.LongTensor(Ctarget_valid).to(device)
#print("C",Ctarget_validTensor )
X_allTensor = torch.FloatTensor(data).to(device)
C_allTensor = torch.LongTensor(data_c).to(device)


train_dataset = TensorDataset(Xtarget_trainTensor, Ctarget_trainTensor)
valid_dataset = TensorDataset(Xtarget_validTensor, Ctarget_validTensor)

Xtarget_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
Xtarget_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

dataloaders_pretrain = {'train':Xtarget_trainDataLoader,'val':Xtarget_validDataLoader}
#print('START SECTION OF LOADING SC DATA TO THE TENSORS')
################################################# START SECTION OF LOADING SC DATA TO THE TENSORS #################################################

################################################# START SECTION OF LOADING BULK DATA  #################################################
# Read source data
data_r=pd.read_csv(source_data_path,index_col=0)
label_r=pd.read_csv(label_path,index_col=0)
label_r=label_r.fillna(na)

# Extract labels
selected_idx = label_r.loc[:,select_drug]!=na
label = label_r.loc[selected_idx.index,select_drug]
data_r = data_r.loc[selected_idx.index,:]
label = label.values.reshape(-1,1)


le = preprocessing.LabelEncoder()
label = le.fit_transform(label)
dim_model_out = 2

# Process source data
mmscaler = preprocessing.MinMaxScaler()
source_data = mmscaler.fit_transform(data_r)

# Split source data
Xsource_train_all, Xsource_test, Ysource_train_all, Ysource_test = train_test_split(source_data,label, test_size=test_size, random_state=42)
Xsource_train, Xsource_valid, Ysource_train, Ysource_valid = train_test_split(Xsource_train_all,Ysource_train_all, test_size=valid_size, random_state=42)

# Transform source data
# Construct datasets and data loaders
Xsource_trainTensor = torch.FloatTensor(Xsource_train).to(device)
Xsource_validTensor = torch.FloatTensor(Xsource_valid).to(device)

Ysource_trainTensor = torch.LongTensor(Ysource_train).to(device)
Ysource_validTensor = torch.LongTensor(Ysource_valid).to(device)

sourcetrain_dataset = TensorDataset(Xsource_trainTensor, Ysource_trainTensor)
sourcevalid_dataset = TensorDataset(Xsource_validTensor, Ysource_validTensor)


Xsource_trainDataLoader = DataLoader(dataset=sourcetrain_dataset, batch_size=batch_size, shuffle=True)
Xsource_validDataLoader = DataLoader(dataset=sourcevalid_dataset, batch_size=batch_size, shuffle=True)

dataloaders_source = {'train':Xsource_trainDataLoader,'val':Xsource_validDataLoader}
#print('END SECTION OF LOADING BULK DATA')
################################################# END SECTION OF LOADING BULK DATA  #################################################


# In[146]:


args.dropout


# In[147]:


################################################# START SECTION OF MODEL CUNSTRUCTION  #################################################
# Construct target encoder
if reduce_model == "AE":
    encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
    loss_function_e = nn.MSELoss()
elif reduce_model == "VAE":
    encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
if reduce_model == "DAE":
    encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
    loss_function_e = nn.MSELoss()        


# In[148]:


dim_model_out = 2
# Load AE model
if reduce_model == "AE":
    source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
            hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
            pretrained_weights=None,freezed=freeze,drop_out=args.dropout,drop_out_predictor=args.dropout)

    #source_model.load_state_dict(torch.load(selected_model))
    source_encoder = source_model
if reduce_model == "DAE":
    source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
            hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
            pretrained_weights=None,freezed=freeze,drop_out=args.dropout,drop_out_predictor=args.dropout)

    #source_model.load_state_dict(torch.load(selected_model))
    source_encoder = source_model    
# Load VAE model
elif reduce_model in ["VAE"]:
    source_model = PretrainedVAEPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
            hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
            pretrained_weights=None,freezed=freeze,z_reparam=bool(args.VAErepram),drop_out=args.dropout,drop_out_predictor=args.dropout)
    #source_model.load_state_dict(torch.load(selected_model))
    source_encoder = source_model


# In[149]:


# Set DaNN model
DaNN_model = DaNN(source_model=source_encoder,target_model=encoder)
DaNN_model.to(device)


# In[150]:


DaNN_model.load_state_dict(torch.load(selected_model))


# In[151]:


encoder = DaNN_model.target_model
source_model = DaNN_model.source_model


# In[152]:


embedding_tensors = encoder.encode(X_allTensor)
embeddings = embedding_tensors.detach().cpu().numpy()


# In[153]:


if(args.target_data!="GSE117872"):
    adata_path = glob.glob("D://pyws//trainsource//saved//adata//review//*"+args.target_data+"*")
else:
    adata_path = glob.glob("D://pyws//trainsource//saved//adata//review//*"+args.batch_id+"*")


# In[154]:


adata_final = sc.read_h5ad(adata_path[0])


# In[155]:


adata_final.obsm["X_Trans"] = embeddings


# In[156]:


data_name


# In[158]:


color_list = ["leiden","sensitivity",'sens_preds']
title_list = ['Cluster',"Ground truth","Probability"]
color_score_list = color_list

# Run embeddings using transfered embeddings
sc.pp.neighbors(adata_final,use_rep='X_Trans',key_added="Trans")
sc.tl.umap(adata_final,neighbors_key="Trans")
sc.tl.leiden(adata_final,neighbors_key="Trans",key_added="leiden_trans",resolution=leiden_res)
sc.pl.umap(adata_final,color=color_list,neighbors_key="Trans",
           save=paras[0]+args.transfer+args.dimreduce+"_TL.tiff",
           show=False,title=title_list)
# Plot cell score on umap
#sc.pl.umap(adata,color=color_score_list,neighbors_key="Trans",save=data_name+args.transfer+args.dimreduce+"_score_TL"+now,show=False,title=color_score_list)

