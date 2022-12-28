#!/usr/bin/env python
# coding: utf-8
import argparse
import pandas as pd
from pandas.core.frame import DataFrame
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import DaNN.mmd as mmd
import scanpypip.preprocessing as pp
import trainers as t
import utils as ut
from captum.attr import IntegratedGradients
from models import (AEBase, DaNN, PretrainedPredictor,
                    PretrainedVAEPredictor, VAEBase)
from scipy.spatial import distance_matrix, minkowski_distance, distance
import networkx as nx
from igraph import *
import random
seed = 42
torch.manual_seed(seed)
#np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#from transformers import *
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

DATA_MAP={
"GSE117872":"data/GSE117872/GSE117872_good_Data_TPM.txt",
"GSE110894":"data/GSE110894/GSE110894.csv",
"GSE112274":"data/GSE112274/GSE112274_cell_gene_FPKM.csv",
"GSE140440":"data/GSE140440/GSE140440.csv",
"GSE149383":"data/GSE149383/erl_total_data_2K.csv",
"GSE110894_small":"data/GSE110894/GSE110894_small.h5ad"
}
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
    
from sklearn.metrics.pairwise import cosine_similarity
### loss2
import copy
def train_DaNN_model2(net,source_loader,target_loader,
                    optimizer,loss_function,n_epochs,scheduler,dist_loss,weight=0.25,GAMMA=1000,epoch_tail=0.90,
                    load=False,save_path="save/model.pkl",best_model_cache = "drive",top_models=5,k=10):

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
                encoderrep = net.target_model.encoder(x_tar)
                #print(x_tar.shape)
                if encoderrep.shape[0]<k:
                    next
                else:    
                    edgeList = calculateKNNgraphDistanceMatrix(encoderrep.cpu().detach().numpy(), distanceType='euclidean', k=10)
                    listResult, size = generateLouvainCluster(edgeList)
                    # sc sim loss
                    loss_s = 0
                    for i in range(size):
                        #print(i)
                        s = cosine_similarity(x_tar[np.asarray(listResult) == i,:].cpu().detach().numpy())
                        s = 1-s
                        loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                    #loss_s = torch.tensor(loss_s).cuda()
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
    
    
def run_main(args):
################################################# START SECTION OF LOADING PARAMETERS #################################################
    # Read parameters

    t0 = time.time()

    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    na = args.missing_value
    if args.sc_data=='GSE117872_HN120':
        data_path = DATA_MAP['GSE117872']
    elif args.sc_data=='GSE117872_HN137':
        data_path = DATA_MAP['GSE117872']
    elif args.sc_data in DATA_MAP:
        data_path = DATA_MAP[args.sc_data]
    else:
        data_path = args.sc_data
    test_size = args.test_size
    select_drug = args.drug.upper()
    freeze = args.freeze_pretrain
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    min_n_genes = args.min_n_genes
    max_n_genes = args.max_n_genes
    
    #print(target_model_path)
    log_path = args.logging_file
    batch_size = args.batch_size
    encoder_hdims = args.bulk_h_dims.split(",")
    encoder_hdims = list(map(int, encoder_hdims))
    
    
    data_name = args.sc_data
    label_path = args.label
    reduce_model = args.dimreduce
    predict_hdims = args.predictor_h_dims.split(",")
    predict_hdims = list(map(int, predict_hdims))
    leiden_res = args.cluster_res
    load_model = bool(args.load_sc_model)
    mod = args.mod
    
    para = str(args.bulk)+"_data_"+str(args.sc_data)+"_drug_"+str(args.drug)+"_bottle_"+str(args.bottleneck)+"_edim_"+str(args.bulk_h_dims)+"_pdim_"+str(args.predictor_h_dims)+"_model_"+reduce_model+"_dropout_"+str(args.dropout)+"_gene_"+str(args.printgene)+"_lr_"+str(args.lr)+"_mod_"+str(args.mod)+"_sam_"+str(args.sampling)    
    source_data_path = args.bulk_data
    pretrain = args.pretrain+para
    source_model_path = args.bulk_model_path+para
    print(source_model_path)
    #print(source_model_path)
    target_model_path = args.sc_model_path +para
    
    # Misc
    now=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Initialize logging and std out
    out_path = log_path+now+"transfer.err"
    log_path = log_path+now+"transfer.log"

    out=open(out_path,"w")
    sys.stderr=out
    
    #Logging infomaion
    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(args)
    logging.info("Start at " + str(t0))

    
    for path in [args.logging_file,args.bulk_model_path,args.sc_model_path,args.pretrain,"save/adata/"]:
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
    
    # Save arguments
    args_df = ut.save_arguments(args,now)
################################################# END SECTION OF LOADING PARAMETERS #################################################

################################################# START SECTION OF SINGLE CELL DATA REPROCESSING #################################################
    # Load data and preprocessing
    adata = pp.read_sc_file(data_path)
    
    if data_name == 'GSE117872_HN137':
        adata =  ut.specific_process(adata,dataname='GSE117872',select_origin='HN137')
    elif data_name == 'GSE117872_HN120':    
        adata =  ut.specific_process(adata,dataname='GSE117872',select_origin='HN120')
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
        adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=args.min_c,
                            filter_mingenes=args.min_g,normalize=True,log=True)
    else:
        adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=args.min_c,percent_mito = args.percent_mito,
                            filter_mingenes=args.min_g,normalize=True,log=True)


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

################################################# END SECTION OF SINGLE CELL DATA REPROCESSING #################################################

################################################# START SECTION OF LOADING SC DATA TO THE TENSORS #################################################
    #Prepare to normailize and split target data
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
    Xtarget_train, Xtarget_valid, Ctarget_train, Ctarget_valid = train_test_split(data,data_c, test_size=valid_size, random_state=42)


    # Select the device of gpu
    if(args.device == "gpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)

    else:
        device = 'cpu'
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    #logging.info(device)
    print(device)

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
    if args.bulk == 'old':
        data_r=data_r[0:805]
        label_r=label_r[0:805]
    elif args.bulk == 'new':
        data_r=data_r[805:data_r.shape[0]]
        label_r=label_r[805:label_r.shape[0]]              
    else:
        print("two databases combine")
    #label_r=label_r.fillna(na)

    # Extract labels
    selected_idx = label_r.loc[:,select_drug]!=na
    label = label_r.loc[selected_idx,select_drug]
    data_r = data_r.loc[selected_idx,:]
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

    #if torch.cuda.is_available():
    #    encoder.cuda()

    #logging.info("Target encoder structure is: ")
    #logging.info(encoder)

    encoder.to(device)
    optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
    loss_function_e = nn.MSELoss()
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)


    dim_model_out = 2
    # Load AE model
    if reduce_model == "AE":
        source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze,drop_out=args.dropout,drop_out_predictor=args.dropout)
        
        source_model.load_state_dict(torch.load(source_model_path))
        source_encoder = source_model
    if reduce_model == "DAE":
        source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze,drop_out=args.dropout,drop_out_predictor=args.dropout)
        
        source_model.load_state_dict(torch.load(source_model_path))
        source_encoder = source_model    
    # Load VAE model
    elif reduce_model in ["VAE"]:
        source_model = PretrainedVAEPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze,z_reparam=bool(args.VAErepram),drop_out=args.dropout,drop_out_predictor=args.dropout)
        source_model.load_state_dict(torch.load(source_model_path))
        source_encoder = source_model
    #logging.info("Load pretrained source model from: "+source_model_path)
           
    source_encoder.to(device)
################################################# END SECTION OF MODEL CUNSTRUCTION  #################################################

################################################# START SECTION OF SC MODEL PRETRAININIG  #################################################
    # Pretrain target encoder
    # Pretain using autoencoder is pretrain is not False
    if(str(pretrain)!='0'):
        # Pretrained target encoder if there are not stored files in the harddisk
        train_flag = True
        pretrain = str(pretrain)
        print(pretrain)
        '''
        if(os.path.exists(pretrain)==True):
            try:
                encoder.load_state_dict(torch.load(pretrain))
                #logging.info("Load pretrained target encoder from "+pretrain)
                train_flag = False

            except:
                print("Loading failed, procceed to re-train model")
        '''      
        if train_flag == True:

            if reduce_model == "AE":
                encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                            optimizer=optimizer_e,loss_function=loss_function_e,
                                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)
            if reduce_model == "DAE":
                encoder,loss_report_en = t.train_DAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                            optimizer=optimizer_e,loss_function=loss_function_e,
                                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)
                                            
            elif reduce_model == "VAE":
                encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                optimizer=optimizer_e,
                                n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)
            #print(loss_report_en)
            #logging.info("Pretrained finished")

        # Before Transfer learning, we test the performance of using no transfer performance:
        # Use vae result to predict 
        embeddings_pretrain = encoder.encode(X_allTensor)
        print(embeddings_pretrain)
        pretrain_prob_prediction = source_model.predict(embeddings_pretrain).detach().cpu().numpy()
        adata.obs["sens_preds_pret"] = pretrain_prob_prediction[:,1]
        adata.obs["sens_label_pret"] = pretrain_prob_prediction.argmax(axis=1)

        # Add embeddings to the adata object
        embeddings_pretrain = embeddings_pretrain.detach().cpu().numpy()
        adata.obsm["X_pre"] = embeddings_pretrain
################################################# END SECTION OF SC MODEL PRETRAININIG  #################################################

################################################# START SECTION OF TRANSFER LEARNING TRAINING #################################################
    # Using ADDA transfer learning
    # DaNN model
    # Set predictor loss
    #print('1',mod)
    loss_d = nn.CrossEntropyLoss()
    optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
    exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)
       
    # Set DaNN model
    #DaNN_model = DaNN(source_model=source_encoder,target_model=encoder)
    DaNN_model = DaNN(source_model=source_encoder,target_model=encoder,fix_source=bool(args.fix_source))
    DaNN_model.to(device)

    def loss(x,y,GAMMA=args.mmd_GAMMA):
        result = mmd.mmd_loss(x,y,GAMMA)
        return result

    loss_disrtibution = loss
    #print('2',mod)
    

     
       # Tran DaNN model
    if mod == 'ori':
        #print('3',DaNN_model) 
        #print(dataloaders_source)     
        #print(dataloaders_pretrain)     
        DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                            dataloaders_source,dataloaders_pretrain,
                            # Should here be all optimizer d?
                            optimizer_d, loss_d,
                            epochs,exp_lr_scheduler_d,
                            dist_loss=loss_disrtibution,
                            load=load_model,
                            weight = args.mmd_weight,
                            save_path=target_model_path+"_DaNN.pkl")
                               
    if mod == 'new': 
        print(mod)
        target_model = TargetModel(source_model,encoder)
        DaNN_model, report_, _, _ = train_DaNN_model2(DaNN_model,
                            dataloaders_source,dataloaders_pretrain,
                            # Should here be all optimizer d?
                            optimizer_d, loss_d,
                            epochs,exp_lr_scheduler_d,
                            dist_loss=loss_disrtibution,
                            load=load_model,
                            weight = args.mmd_weight,
                            save_path=target_model_path+"_DaNN.pkl")                        

    encoder = DaNN_model.target_model
    source_model = DaNN_model.source_model

    #logging.info("Transfer DaNN finished")
################################################# END SECTION OF TRANSER LEARNING TRAINING #################################################


################################################# START SECTION OF PREPROCESSING FEATURES #################################################
    # Extract feature embeddings 
    # Extract prediction probabilities

    embedding_tensors = encoder.encode(X_allTensor)
    prediction_tensors = source_model.predictor(embedding_tensors)
    embeddings = embedding_tensors.detach().cpu().numpy()
    predictions = prediction_tensors.detach().cpu().numpy()
    print("predictions",predictions.shape)
    # Transform predict8ion probabilities to 0-1 labels

    adata.obs["sens_preds"] = predictions[:,1]
    adata.obs["sens_label"] = predictions.argmax(axis=1)
    adata.obs["sens_label"] = adata.obs["sens_label"].astype('category')
    adata.obs["rest_preds"] = predictions[:,0]
    
################################################# END SECTION OF PREPROCESSING FEATURES #################################################
    
    
    
    
    #sc_critical_gene = adata.var.index[np.mean(abs(attr),0)> np.percentile(np.mean(abs(attr),0),95)]
    

################################################# START SECTION OF ANALYSIS AND POST PROCESSING #################################################
################################################# END SECTION OF ANALYSIS AND POST PROCESSING #################################################

################################################# START SECTION OF ANALYSIS FOR BULK DATA #################################################
    # Save adata
  
    adata.write("save/adata/"+data_name+para+".h5ad")
################################################# END SECTION OF ANALYSIS FOR BULK DATA #################################################
    from sklearn.metrics import (average_precision_score,
                             classification_report, mean_squared_error, r2_score, roc_auc_score)
    sens_pb_pret = adata.obs['sens_preds_pret']
    lb_pret = adata.obs['sens_label_pret']
    report_df = {}
    Y_test = adata.obs['sensitive']
    sens_pb_results = adata.obs['sens_preds']
    lb_results = adata.obs['sens_label']
    
    #Y_test ture label
    ap_score = average_precision_score(Y_test, sens_pb_results)
    ap_pret = average_precision_score(Y_test, sens_pb_pret)
    # ap_umap = average_precision_score(Y_test, sens_pb_umap)
    # ap_tsne = average_precision_score(Y_test, sens_pb_tsne)
    
    
    report_dict = classification_report(Y_test, lb_results, output_dict=True)
    f1score = report_dict['weighted avg']['f1-score']
    report_df['f1_score'] = f1score
    file = 'save/bulk_f'+data_name+'_f1_score_ori.txt'
    with open(file, 'a+') as f:
         f.write(para+'\t'+str(f1score)+'\n') 
    print("sc model finished")
    if (args.printgene=='T'):
        target_model = TargetModel(source_model,encoder)
        sc_X_allTensor=X_allTensor
        
        ytarget_allPred = target_model(sc_X_allTensor).detach().cpu().numpy()
        ytarget_allPred = ytarget_allPred.argmax(axis=1)
        # Caculate integrated gradient
        ig = IntegratedGradients(target_model)
        
        
        scattr, delta =  ig.attribute(sc_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        scattr = scattr.detach().cpu().numpy()
        
        igadata= sc.AnnData(scattr)
        igadata.var.index = adata.var.index
        igadata.obs.index = adata.obs.index
        sc_gra = "ori_result/" + data_name +"sc_gradient.txt"
        #sc_exp = "result/" + data_name +"sc_expression.txt"
        sc_gen = "ori_result/" + data_name +"sc_gene.csv"
        sc_lab = "ori_result/" + data_name +"sc_label.csv"
        #np.savetxt("result/sc_embedding.txt",(sc_X_allTensor).detach().cpu().numpy(),delimiter = " ")
        #np.savetxt(sc_exp,adata.X,delimiter = " ")
        np.savetxt(sc_gra,scattr,delimiter = " ")
        DataFrame(adata.var.index).to_csv(sc_gen)
        DataFrame(adata.obs["sens_label"]).to_csv(sc_lab)
    t1 = time.time()

    #logging.info("End at " + str(t1)+", takes :" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--bulk_data', type=str, default='data/ALL_expression.csv',help='Path of the bulk RNA-Seq expression profile')
    parser.add_argument('--label', type=str, default='data/ALL_label_binary_wf.csv',help='Path of the processed bulk RNA-Seq drug screening annotation')
    parser.add_argument('--sc_data', type=str, default="GSE110894",help='Accession id for testing data, only support pre-built data.')
    parser.add_argument('--drug', type=str, default='I-BET-762',help='Name of the selected drug, should be a column name in the input file of --label')
    parser.add_argument('--missing_value', type=int, default=1,help='The value filled in the missing entry in the drug screening annotation, default: 1')
    parser.add_argument('--test_size', type=float, default=0.2,help='Size of the test set for the bulk model traning, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,help='Size of the validation set for the bulk model traning, default: 0.2')
    parser.add_argument('--var_genes_disp', type=float, default=0,help='Dispersion of highly variable genes selection when pre-processing the data. \
                         If None, all genes will be selected .default: None')
    parser.add_argument('--min_n_genes', type=int, default=0,help="Minimum number of genes for a cell that have UMI counts >1 for filtering propose, default: 0 ")
    parser.add_argument('--max_n_genes', type=int, default=20000,help="Maximum number of genes for a cell that have UMI counts >1 for filtering propose, default: 20000 ")
    parser.add_argument('--min_g', type=int, default=200,help="Minimum number of genes for a cell >1 for filtering propose, default: 200")
    parser.add_argument('--min_c', type=int, default=3,help="Minimum number of cell that each gene express for filtering propose, default: 3")
    parser.add_argument('--percent_mito', type=int, default=100,help="Percentage of expreesion level of moticondrial genes of a cell for filtering propose, default: 100")

    parser.add_argument('--cluster_res', type=float, default=0.2,help="Resolution of Leiden clustering of scRNA-Seq data, default: 0.3")
    parser.add_argument('--mmd_weight', type=float, default=0.25,help="Weight of the MMD loss of the transfer learning, default: 0.25")
    parser.add_argument('--mmd_GAMMA', type=int, default=1000,help="Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000")

    # train
    parser.add_argument('--device', type=str, default="cpu",help='Device to train the model. Can be cpu or gpu. Deafult: cpu')
    parser.add_argument('--bulk_model_path','-s', type=str, default='save/bulk_pre/',help='Path of the trained predictor in the bulk level')
    parser.add_argument('--sc_model_path', '-p',  type=str, default='save/sc_pre/',help='Path (prefix) of the trained predictor in the single cell level')
    parser.add_argument('--pretrain', type=str, default='save/sc_encoder/',help='Path of the pre-trained encoder in the single-cell level')

    parser.add_argument('--lr', type=float, default=1e-2,help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=500,help='Number of epoches training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=200,help='Number of batch size when training. Default: 200')
    parser.add_argument('--bottleneck', type=int, default=512,help='Size of the bottleneck layer of the model. Default: 32')
    parser.add_argument('--dimreduce', type=str, default="AE",help='Encoder model type. Can be AE or VAE. Default: AE')
    parser.add_argument('--freeze_pretrain', type=int,default=0,help='Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0')
    parser.add_argument('--bulk_h_dims', type=str, default="512,256",help='Shape of the source encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--sc_h_dims', type=str, default="512,256",help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--predictor_h_dims', type=str, default="16,8",help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 16,8')
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--batch_id', type=str, default="HN137",help="Batch id only for testing")
    parser.add_argument('--load_sc_model', type=int, default=0,help='Load a trained model or not. 0: do not load, 1: load. Default: 0')
    
    parser.add_argument('--mod', type=str, default='new',help='Embed the cell type label to regularized the training: new: add cell type info, ori: do not add cell type info. Default: new')
    parser.add_argument('--printgene', type=str, default='T',help='Print the cirtical gene list: T: print. Default: T')
    parser.add_argument('--dropout', type=float, default=0.3,help='Dropout of neural network. Default: 0.3')
    # miss
    parser.add_argument('--logging_file', '-l',  type=str, default='save/logs/',help='Path of training log')
    parser.add_argument('--sampling', type=str, default=None,help='Samping method of training data for the bulk model traning. \
                        Can be upsampling, downsampling, or SMOTE. default: None')
    parser.add_argument('--fix_source', type=int, default=0,help='Fix the bulk level model. Default: 0')
    parser.add_argument('--bulk', type=str, default='integrate',help='Selection of the bulk database.integrate:both dataset. old: GDSC. new: CCLE. Default: integrate')
    #
    args, unknown = parser.parse_known_args()
    run_main(args)
