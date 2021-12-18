import torch
import numpy as np
import random
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
### loss2
import copy

import logging
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score,
                             classification_report, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import  nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

import sampling as sam
import utils as ut
import trainers as t
import matplotlib



import scanpy as sc
import DaNN.mmd as mmd
import scanpypip.preprocessing as pp
from models import (AEBase, DaNN, PretrainedPredictor,
                    PretrainedVAEPredictor, VAEBase)


from captum.attr import IntegratedGradients
from pandas.core.frame import DataFrame
seed=0
#random.seed(seed)
torch.manual_seed(seed)
#np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#from transformers import *
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
from scipy.spatial import distance_matrix, minkowski_distance, distance
import networkx as nx
from igraph import *
parser = argparse.ArgumentParser()
# data 
parser.add_argument('--select_drug', type=str, default='Cisplatin',help='select_drug')
parser.add_argument('--data_name', type=str, default='GSE117872',help='data_name')
parser.add_argument('--encoder_hdims', type=str, default="512,256",help='encoder_hdims')
parser.add_argument('--preditor_hdims', type=str, default="64,32",help='preditor_hdims')
parser.add_argument('--reduce_model', type=str, default='AE',help='reduce_model')
parser.add_argument('--dim_au_out', type=int, default=128,help='bottle_neck')
parser.add_argument('--mod', type=str, default='ori',help='mod')
warnings.filterwarnings("ignore")
#para="data_name_"+data_name+"_encoder_hdims_"+args.encoder_hdims+"_preditor_hdims_"+args.preditor_hdims+"_reduce_model_"+reduce_model+"_bottle_neck_"+str(dim_au_out)+"_mod_"+args.mod
#print(para)
args, unknown = parser.parse_known_args()
select_drug = args.select_drug
data_name =  args.data_name
args_sc_data = data_name
encoder_hdims = args.preditor_hdims.split(",")
preditor_hdims = args.preditor_hdims.split(",")
reduce_model = args.reduce_model
dim_au_out = args.dim_au_out #bottle-neck
mod = args.mod
print(data_name)


#log_path = args.logging_file
batch_size = 200
#encoder_hdims = "512,256".split(",")
#encoder_hdims = list(map(int, encoder_hdims))
#source_data_path = "data/GDSC2_expression.csv" 

dim_model_out = 2
data_path = 'data/GDSC2_expression.csv'
label_path = "data/GDSC2_label_9drugs_binary.csv"
#model_path = "model/" + data_name + "_predictor_bulk.pkl" 
bulk_encoder = "model/" + data_name + "_encoder_bulk.pkl"
epochs = 500

na = 1
VAErepram=1
test_size = 0.2
valid_size = 0.2
g_disperson = None
freeze_pretrain = 0



sampling = None
PCA_dim = 0
load_source_model=0
#dim_model_out = 128
test_size = 0.2
valid_size = 0.2
encoder_hdims = list(map(int, encoder_hdims) )
preditor_hdims = list(map(int, preditor_hdims) )
load_model = bool(load_source_model)


min_n_genes = 0
max_n_genes = 20000
para="DN_"+data_name+"_EH_"+args.encoder_hdims+"_PH_"+args.preditor_hdims+"_RM_"+reduce_model+"_BN_"+str(dim_au_out)+"_mod_"+args.mod
print(para)
target_model_path = "/users/PAS1475/anjunma/wxy/scDEAL/model/"+para+"predictor_sc" 
source_model_path =  "/users/PAS1475/anjunma/wxy/scDEAL/model/"+para+"_predictor_bulk.pkl"
pretrain =  "/users/PAS1475/anjunma/wxy/scDEAL/model/"+para+"_encoder_sc.pkl"
model_path = source_model_path
#pretrain =  "/users/PAS1475/anjunma/wxy/scDEAL/model/"+data_name+"_encoder_sc.pkl"


leiden_res = 0.3

min_g = 200
min_c = 10
percent_mito =  100
#cluster_Res = 0.3
mmd_weight = 0.25
mmd_GAMMA = 1000



class TargetModel(nn.Module):
    def __init__(self, source_predcitor,target_encoder):
        super(TargetModel, self).__init__()
        self.source_predcitor = source_predcitor
        self.target_encoder = target_encoder

    def forward(self, X_target,C_target=None):

        if(type(C_target)==type(None)):
            x_tar = self.target_encoder.encode(X_target)
        else:
            x_tar = self.target_encoder.encode(X_target,C_target)
        y_src = self.source_predcitor.predictor(x_tar)
        return y_src

from captum.attr import IntegratedGradients
def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=10):
    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
    #print(distMat)
    edgeList=[]

    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append((i,res[j],distMat[i,j]))
    
    return edgeList

def generateLouvainCluster(edgeList):
   
    """
    Louvain Clustering using igraph
    """
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    W = nx.adjacency_matrix(Gtmp)
    W = W.todense()
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)
    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)
    size = len(louvain_partition)
    hdict = {}
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = []
    for i in range(count):
        listResult.append(hdict[i])

    return listResult, size


def train_DaNN_model2(net,source_loader,target_loader,
                    optimizer,loss_function,n_epochs,scheduler,dist_loss,weight=0.25,GAMMA=1000,epoch_tail=0.90,
                    load=False,save_path="saved/model.pkl",best_model_cache = "drive",top_models=5):

    if(load!=False):
        if(os.path.exists(save_path)):
            try:
                net.load_state_dict(torch.load(save_path))           
                return net, 0
            except:
                logging.warning("Failed to load existing file, proceed to the trainning process.")

        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: source_loader[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    mmd_train = {}
    sc_train = {}
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf


    g_tar_outputs = []
    g_src_outputs = []

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_mmd = 0.0
            running_sc =0.0
            
            batch_j = 0
            list_src, list_tar = list(enumerate(source_loader[phase])), list(enumerate(target_loader[phase]))
            n_iters = max(len(source_loader[phase]), len(target_loader[phase]))

            for batchidx, (x_src, y_src) in enumerate(source_loader[phase]):
                _, (x_tar, y_tar) = list_tar[batch_j]
                
                x_tar.requires_grad_(True)
                x_src.requires_grad_(True)

                min_size = min(x_src.shape[0],x_tar.shape[0])

                if (x_src.shape[0]!=x_tar.shape[0]):
                    x_src = x_src[:min_size,]
                    y_src = y_src[:min_size,]
                    x_tar = x_tar[:min_size,]
                    y_tar = y_tar[:min_size,]

                #x.requires_grad_(True)
                # encode and decode 
                
                
                
                if(net.target_model._get_name()=="CVAEBase"):
                    y_pre, x_src_mmd, x_tar_mmd = net(x_src, x_tar,y_tar)
                else:
                    y_pre, x_src_mmd, x_tar_mmd = net(x_src, x_tar)
                # compute loss
                encoderrep = target_model.target_encoder.encoder(x_tar)
                #print(x_tar.shape)
                edgeList = calculateKNNgraphDistanceMatrix(encoderrep.cpu().detach().numpy(), distanceType='euclidean', k=10)
                listResult, size = generateLouvainCluster(edgeList)
                # sc sim loss
                loss_s = 0
                for i in range(size):
                    #print(i)
                    s = cosine_similarity(x_tar[np.asarray(listResult) == i,:].cpu().detach().numpy())
                    s = 1-s
                    loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                loss_s = torch.tensor(loss_s).cuda()
                loss_s.requires_grad_(True)
                loss_c = loss_function(y_pre, y_src)      
                loss_mmd = dist_loss(x_src_mmd, x_tar_mmd)
                #print(loss_s,loss_c,loss_mmd)

                loss = loss_c + weight * loss_mmd +loss_s


                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward(retain_graph=True)
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
                running_mmd += loss_mmd.item()
                running_sc += loss_s.item()
                # Iterate over batch
                batch_j += 1
                if batch_j >= len(list_tar):
                    batch_j = 0

            # Average epoch loss
            epoch_loss = running_loss / n_iters
            epoch_mmd = running_mmd/n_iters
            epoch_sc = running_sc/n_iters
            # Step schedular
            if phase == 'train':
                scheduler.step(epoch_loss)
            
            # Savle loss
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            mmd_train[epoch,phase] = epoch_mmd
            sc_train[epoch,phase] = epoch_sc
            
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if (phase == 'val') and (epoch_loss < best_loss) and (epoch >(n_epochs*(1-epoch_tail))) :
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(net.state_dict())
                # Save model if acheive better validation score
                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
 
    #     # Select best model wts
    #     torch.save(best_model_wts, save_path)
        
    # net.load_state_dict(best_model_wts)           
        # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    return net, loss_train,mmd_train,sc_train





#########bulk data
# pretrain bulk predict model path
#preditor_path = model_path + reduce_model  + select_drug + '.pkl'
data_r=pd.read_csv(data_path,index_col=0)
label_r=pd.read_csv(label_path,index_col=0)
label_r=label_r.fillna(na)
# Filter out na values
selected_idx = label_r.loc[:,select_drug]!=na
# highly_variable_gene 

if(g_disperson!=None):
        hvg,adata = ut.highly_variable_genes(data_r,min_disp=g_disperson)
        # Rename columns if duplication exist
        data_r.columns = adata.var_names
        # Extract hvgs
        data = data_r.loc[selected_idx.index,hvg]
else:
        data = data_r.loc[selected_idx.index,:]
PCA_dim = 0  

if PCA_dim !=0 :
        data = PCA(n_components = PCA_dim).fit_transform(data)
else:
        data = data        
label = label_r.loc[selected_idx.index,select_drug]
data_r = data_r.loc[selected_idx.index,:]

# Scaling data
mmscaler = preprocessing.MinMaxScaler()

data = mmscaler.fit_transform(data)
label = label.values.reshape(-1,1)


le = LabelEncoder()
label = le.fit_transform(label)
X_train_all, bulk_X_test, Y_train_all, bulk_Y_test = train_test_split(data, label, test_size=test_size, random_state=42)
bulk_X_train, bulk_X_valid, bulk_Y_train, bulk_Y_valid = train_test_split(X_train_all, Y_train_all, test_size=valid_size, random_state=42)
# sampling method
if sampling == None:
    X_train,Y_train=sam.nosampling(bulk_X_train,bulk_Y_train)
    #logging.info("nosampling")
elif sampling =="upsampling":
    X_train,Y_train=sam.upsampling(bulk_X_train,bulk_Y_train)
    #logging.info("upsampling")
elif sampling =="downsampling":
    X_train,Y_train=sam.downsampling(bulk_X_train,bulk_Y_train)
    #logging.info("downsampling")
elif  sampling=="SMOTE":
    X_train,Y_train=sam.SMOTEsampling(X_train,bulk_Y_train)
    #logging.info("SMOTE")
else:
    print("error")
    #logging.info("not a legal sampling method")
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
# Construct datasets and data loaders
bulk_X_trainTensor = torch.FloatTensor(bulk_X_train).to(device)
bulk_X_validTensor = torch.FloatTensor(bulk_X_valid).to(device)
bulk_X_testTensor = torch.FloatTensor(bulk_X_test).to(device)

bulk_Y_trainTensor = torch.LongTensor(bulk_Y_train).to(device)
bulk_Y_validTensor = torch.LongTensor(bulk_Y_valid).to(device)

# Preprocess data to tensor
bulk_train_dataset = TensorDataset(bulk_X_trainTensor, bulk_X_trainTensor)
bulk_valid_dataset = TensorDataset(bulk_X_validTensor, bulk_X_validTensor)

bulk_X_trainDataLoader = DataLoader(dataset=bulk_train_dataset, batch_size=batch_size, shuffle=True)
bulk_X_validDataLoader = DataLoader(dataset=bulk_valid_dataset, batch_size=batch_size, shuffle=True)


bulk_X_allTensor = torch.FloatTensor(data).to(device)
bulk_Y_allTensor = torch.LongTensor(label).to(device)

# construct TensorDataset
bulk_trainreducedDataset = TensorDataset(bulk_X_trainTensor, bulk_Y_trainTensor)
bulk_validreducedDataset = TensorDataset(bulk_X_validTensor, bulk_Y_validTensor)

bulk_trainDataLoader_p = DataLoader(dataset=bulk_trainreducedDataset, batch_size=batch_size, shuffle=True)
bulk_validDataLoader_p = DataLoader(dataset=bulk_validreducedDataset, batch_size=batch_size, shuffle=True)
bulk_dataloaders_train = {'train':bulk_trainDataLoader_p,'val':bulk_validDataLoader_p}
###bulk autoencoder

args_pretrain = True
if(bool(args_pretrain)!=False):
    bulk_dataloaders_pretrain = {'train':bulk_X_trainDataLoader,'val':bulk_X_validDataLoader}
    if reduce_model == "VAE":
        encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
    if reduce_model == "AE":
        encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)

    if torch.cuda.is_available():
        encoder.cuda()

    #logging.info(encoder)
    encoder.to(device)

    optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
    loss_function_e = nn.MSELoss()
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)
    print("111")
    if reduce_model == "AE":
        encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=bulk_dataloaders_pretrain,
                                    optimizer=optimizer_e,loss_function=loss_function_e,
                                    n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder)
    if reduce_model == "VAE":
        encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=bulk_dataloaders_pretrain,
                        optimizer=optimizer_e,
                        n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder)
## bulk predict        
if reduce_model == "AE":
            bulk_predict_model = PretrainedPredictor(input_dim=bulk_X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                                hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                pretrained_weights=bulk_encoder,freezed=bool(0))   
if reduce_model == "VAE":
    bulk_predict_model = PretrainedVAEPredictor(input_dim=bulk_X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                    hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                    pretrained_weights=bulk_encoder,freezed=bool(0),z_reparam=bool(VAErepram))        

if torch.cuda.is_available():
        bulk_predict_model.cuda()
optimizer = optim.Adam(bulk_predict_model.parameters(), lr=1e-2)

loss_function = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

# Train prediction model
bulk_predict_model,report = t.train_predictor_model(bulk_predict_model,bulk_dataloaders_train,
                                        optimizer,loss_function,epochs,exp_lr_scheduler,load=0,save_path=source_model_path)        
# Evaluation
# embedding [k,2]
bulk_dl_result = bulk_predict_model(bulk_X_testTensor).detach().cpu().numpy()   
# classification  [k,2] 0,1
bulk_lb_results = np.argmax(bulk_dl_result,axis=1)
#pb_results = np.max(dl_result,axis=1)
bulk_pb_results = bulk_dl_result[:,1]

# train bulk predict model evaluation AUC ROC recall F1-score
bulk_report_dict = classification_report(bulk_Y_test, bulk_lb_results, output_dict=True)
report_df = pd.DataFrame(bulk_report_dict).T
ap_score = average_precision_score(bulk_Y_test, bulk_pb_results)
auroc_score = roc_auc_score(bulk_Y_test, bulk_pb_results)
report_df['auroc_score'] = auroc_score
report_df['ap_score'] = ap_score
import scanpypip.utils as ut
bulk_adata = pp.read_sc_file(data_path)
## bulk test predict critical gene

bulk_adata = bulk_adata

bulk_pre = bulk_predict_model(bulk_X_allTensor).detach().cpu().numpy()  
bulk_pre = bulk_pre.argmax(axis=1)
# Caculate integrated gradient
ig = IntegratedGradients(bulk_predict_model)

df_results_p = {}
target=1
attr, delta =  ig.attribute(bulk_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)

#attr, delta =  ig.attribute(bulk_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
attr = attr.detach().cpu().numpy()

bul_exp = "result/" + data_name + "bulk_expression"
bul_gra = "result/" + data_name + "bulk_gradient.txt"
bul_lab = "result/" + data_name + "bulk_lab.csv"
#np.savetxt("result/bulk_embedding.txt",(bulk_X_allTensor).detach().cpu().numpy(),delimiter = " ")
np.savetxt(bul_exp,data_r,delimiter = " ")
np.savetxt(bul_gra,attr,delimiter = " ")
DataFrame(bulk_pre).to_csv(bul_lab)
#########scRNA data


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
"GSE110894_small":"data/GSE110894/GSE110894_small.h5ad"

}


t0 = time.time()
#selsect data


g_disperson = 0


if args_sc_data in DATA_MAP:
    data_path = DATA_MAP[args_sc_data]
else:
    data_path = args_sc_data
import utils as ut


adata = pp.read_sc_file(data_path)
if data_name == 'GSE117872':
    adata =  ut.specific_process(adata,dataname=data_name,select_origin="HN137")
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
    
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = pp.cal_ncount_ngenes(adata)
#Preprocess data by filtering

if data_name not in ['GSE112274','GSE140440']:
    adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=min_c,
                        filter_mingenes=min_g,normalize=True,log=True)
else:
    adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=min_c,percent_mito = percent_mito,
                        filter_mingenes=min_g,normalize=True,log=True)


# Select highly variable genes
sc.pp.highly_variable_genes(adata,min_disp=g_disperson,max_disp=np.inf,max_mean=6)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]

# Preprocess data if spcific process is required
data=adata.X
# PCA
# Generate neighbor graph
sc.tl.pca(adata,svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10)
# Generate cluster labels
sc.tl.leiden(adata,resolution=leiden_res)
sc.tl.umap(adata)
adata.obs['leiden_origin']= adata.obs['leiden']
adata.obsm['X_umap_origin']= adata.obsm['X_umap']
data_c = adata.obs['leiden'].astype("long").to_list()   

mmscaler = preprocessing.MinMaxScaler()

try:
    data = mmscaler.fit_transform(data)

except:
    logging.warning("Only one class, no ROC")

    # Process sparse data
    data = data.todense()
    data = mmscaler.fit_transform(data)

# Split data to train and valid set
# Along with the leiden conditions for CVAE propose
sc_Xtarget_train, sc_Xtarget_valid, sc_Ctarget_train, sc_Ctarget_valid = train_test_split(data,data_c, test_size=valid_size, random_state=42)

#################################################

# Split data to train and valid set
# Along with the leiden conditions for CVAE propose
sc_Xtarget_train, sc_Xtarget_valid, sc_Ctarget_train, sc_Ctarget_valid = train_test_split(data,data_c, test_size=valid_size, random_state=42)

#################################################
#Prepare to normailize and split target data


# Select the device of gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
logging.info(device)
torch.cuda.set_device(device)

# Construct datasets and data loaders
sc_Xtarget_trainTensor = torch.FloatTensor(sc_Xtarget_train).to(device)
sc_Xtarget_validTensor = torch.FloatTensor(sc_Xtarget_valid).to(device)

# Use leiden label if CVAE is applied 
sc_Ctarget_trainTensor = torch.LongTensor(sc_Ctarget_train).to(device)
sc_Ctarget_validTensor = torch.LongTensor(sc_Ctarget_valid).to(device)

sc_X_allTensor = torch.FloatTensor(data).to(device)
sc_C_allTensor = torch.LongTensor(data_c).to(device)


sc_train_dataset = TensorDataset(sc_Xtarget_trainTensor, sc_Ctarget_trainTensor)
sc_valid_dataset = TensorDataset(sc_Xtarget_validTensor, sc_Ctarget_validTensor)

sc_Xtarget_trainDataLoader = DataLoader(dataset=sc_train_dataset, batch_size=batch_size, shuffle=True)
sc_Xtarget_validDataLoader = DataLoader(dataset=sc_valid_dataset, batch_size=batch_size, shuffle=True)

sc_dataloaders_pretrain = {'train':sc_Xtarget_trainDataLoader,'val':sc_Xtarget_validDataLoader}

#dim_model_out = 2
# Load bulk AE model
if reduce_model == "AE":
    source_model = PretrainedPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                                hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                pretrained_weights=None,freezed=bool(0))  
    source_model.load_state_dict(torch.load(source_model_path))
    source_encoder = source_model

elif reduce_model =="VAE":
    source_model = PretrainedVAEPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
            hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
            pretrained_weights=None,freezed=bool(0),z_reparam=bool(VAErepram))
    source_model.load_state_dict(torch.load(source_model_path))
    source_encoder = source_model

source_encoder.to(device)

###sc training
if reduce_model == "AE":
    encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
    loss_function_e = nn.MSELoss()
elif reduce_model == "VAE":
    encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)

if torch.cuda.is_available():
    encoder.cuda()

logging.info("Target encoder structure is: ")
logging.info(encoder)

encoder.to(device)
optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
loss_function_e = nn.MSELoss()
exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)
# Using ADDA transfer learning
# DaNN model
# Set predictor loss

#pretrain=1
if(str(pretrain)!='0'):
    # Pretrained target encoder if there are not stored files in the harddisk
    train_flag = True
    pretrain = str(pretrain)
    if(os.path.exists(pretrain)==True):
        try:
            encoder.load_state_dict(torch.load(pretrain))
            logging.info("Load pretrained target encoder from "+pretrain)
            train_flag = False

        except:
            logging.warning("Loading failed, procceed to re-train model")

    if train_flag == True:

        if reduce_model == "AE":
            encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=sc_dataloaders_pretrain,
                                        optimizer=optimizer_e,loss_function=loss_function_e,
                                        n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)
        elif reduce_model == "VAE":
            encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=sc_dataloaders_pretrain,
                            optimizer=optimizer_e,
                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)

        logging.info("Pretrained finished")

    # Before Transfer learning, we test the performance of using no transfer performance:
    # Use vae result to predict 
    embeddings_pretrain = encoder.encode(sc_X_allTensor)

    pretrain_prob_prediction = source_model.predict(embeddings_pretrain).detach().cpu().numpy()
    adata.obs["sens_preds_pret"] = pretrain_prob_prediction[:,1]
    adata.obs["sens_label_pret"] = pretrain_prob_prediction.argmax(axis=1)

    # Add embeddings to the adata object
    embeddings_pretrain = embeddings_pretrain.detach().cpu().numpy()
    adata.obsm["X_pre"] = embeddings_pretrain
    
# Set DaNN model
loss_d = nn.CrossEntropyLoss()
optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)
DaNN_model = DaNN(source_model=source_encoder,target_model=encoder)
DaNN_model.to(device)

def loss(x,y,GAMMA=mmd_GAMMA):
    result = mmd.mmd_loss(x,y,GAMMA)
    return result

loss_disrtibution = loss
#mod = 'ori'
# Tran DaNN model
if mod == 'ori':
    DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                        bulk_dataloaders_train,sc_dataloaders_pretrain,
                        # Should here be all optimizer d?
                        optimizer_d, loss_d,
                        epochs,exp_lr_scheduler_d,
                        dist_loss=loss_disrtibution,
                        load=load_model,
                        weight = mmd_weight,
                        save_path=target_model_path+"_DaNN.pkl")
if mod == 'new': 

    target_model = TargetModel(source_model,encoder)
    DaNN_model, report_, _, _ = train_DaNN_model2(DaNN_model,
                        bulk_dataloaders_train,sc_dataloaders_pretrain,
                        # Should here be all optimizer d?
                        optimizer_d, loss_d,
                        epochs,exp_lr_scheduler_d,
                        dist_loss=loss_disrtibution,
                        load=load_model,
                        weight = mmd_weight,
                        save_path=target_model_path+"_DaNN.pkl")
encoder = DaNN_model.target_model
source_model = DaNN_model.source_model
logging.info("Transfer DaNN finished")

# Extract feature embeddings 
# Extract prediction probabilities

embedding_tensors = encoder.encode(sc_X_allTensor)
prediction_tensors = source_model.predictor(embedding_tensors)
predictions = prediction_tensors.detach().cpu().numpy()

# Transform predict8ion probabilities to 0-1 labels

adata.obs["sens_preds"] = predictions[:,1]
adata.obs["sens_label"] = predictions.argmax(axis=1)
adata.obs["sens_label"] = adata.obs["sens_label"].astype('category')
adata.obs["rest_preds"] = predictions[:,0]

#sc.set_figure_params(dpi=100, color_map = 'viridis_r')
#sc.settings.verbosity = 1
#sc.logging.print_header()
# Add embeddings to the adata package
#adata.obsm["X_Trans"] = embeddings
#sc.tl.umap(adata)
#sc.pp.neighbors(adata, n_neighbors=10,use_rep="X_Trans")
# Use t-sne on transfer learning features
#sc.tl.umap(adata)

sens_pb_pret = adata.obs['sens_preds_pret']
lb_pret = adata.obs['sens_label_pret']
report_df = report_df.T
Y_test = adata.obs['sensitive']
sens_pb_results = adata.obs['sens_preds']
lb_results = adata.obs['sens_label']
le_sc = LabelEncoder()
le_sc.fit(['Resistant','Sensitive'])
###single cell true label
label_descrbie = le_sc.inverse_transform(Y_test)
adata.obs['sens_truth'] = label_descrbie
color_list = ["sens_truth","sens_label",'sens_preds']
color_score_list = ["Sensitive_score","Resistant_score","1_score","0_score"]
sens_score = pearsonr(adata.obs["sens_preds"],adata.obs["Sensitive_score"])[0]
resistant_score = pearsonr(adata.obs["rest_preds"],adata.obs["Resistant_score"])[0]
report_df['prob_sens_pearson'] = sens_score
report_df['prob_rest_pearson'] = resistant_score
#Y_test ture label
ap_score = average_precision_score(Y_test, sens_pb_results)
ap_pret = average_precision_score(Y_test, sens_pb_pret)
# ap_umap = average_precision_score(Y_test, sens_pb_umap)
# ap_tsne = average_precision_score(Y_test, sens_pb_tsne)


report_dict = classification_report(Y_test, lb_results, output_dict=True)
f1score = report_dict['weighted avg']['f1-score']
report_df['f1_score'] = f1score
print(f1score)
para="data_name_"+data_name+"_encoder_hdims_"+args.encoder_hdims+"_preditor_hdims_"+args.preditor_hdims+"_reduce_model_"+reduce_model+"_bottle_neck_"+str(dim_au_out)+"_mod_"+args.mod
print(para)


target_model = TargetModel(source_model,encoder)
ytarget_allPred = target_model(sc_X_allTensor).detach().cpu().numpy()
ytarget_allPred = ytarget_allPred.argmax(axis=1)
# Caculate integrated gradient
ig = IntegratedGradients(target_model)


scattr, delta =  ig.attribute(sc_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
scattr = scattr.detach().cpu().numpy()
igadata= sc.AnnData(scattr)
igadata.var.index = adata.var.index
igadata.obs.index = adata.obs.index
#sc_critical_gene = adata.var.index[np.mean(abs(attr),0)> np.percentile(np.mean(abs(attr),0),95)]
from pandas.core.frame import DataFrame
sc_gra = "result/" + data_name +"sc_gradient.txt"
#sc_exp = "result/" + data_name +"sc_expression.txt"
sc_gen = "result/" + data_name +"sc_gene.csv"
sc_lab = "result/" + data_name +"sc_label.csv"
#np.savetxt("result/sc_embedding.txt",(sc_X_allTensor).detach().cpu().numpy(),delimiter = " ")
#np.savetxt(sc_exp,adata.X,delimiter = " ")
np.savetxt(sc_gra,scattr,delimiter = " ")
DataFrame(adata.var.index).to_csv(sc_gen)
DataFrame(adata.obs["sens_label"]).to_csv(sc_lab)
