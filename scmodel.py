#!/usr/bin/env python
# coding: utf-8
import argparse
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
from models import (AEBase, DaNN, PretrainedPredictor,
                    PretrainedVAEPredictor, VAEBase)

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

def run_main(args):
################################################# START SECTION OF LOADING PARAMETERS #################################################
    # Read parameters

    t0 = time.time()

    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    na = args.missing_value

    if args.sc_data in DATA_MAP:
        data_path = DATA_MAP[args.sc_data]
    else:
        data_path = args.sc_data
    test_size = args.test_size
    select_drug = args.drug
    freeze = args.freeze_pretrain
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    min_n_genes = args.min_n_genes
    max_n_genes = args.max_n_genes
    source_model_path = args.bulk_model_path
    target_model_path = args.sc_model_path 
    log_path = args.logging_file
    batch_size = args.batch_size
    encoder_hdims = args.bulk_h_dims.split(",")
    encoder_hdims = list(map(int, encoder_hdims))
    source_data_path = args.bulk_data 
    pretrain = args.pretrain
    data_name = args.sc_data
    label_path = args.label
    reduce_model = args.dimreduce
    predict_hdims = args.predictor_h_dims.split(",")
    predict_hdims = list(map(int, predict_hdims))
    leiden_res = args.cluster_res
    load_model = bool(args.load_sc_model)

    
    # Misc
    now=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Initialize logging and std out
    out_path = log_path+now+".err"
    log_path = log_path+now+".log"

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

    
    # Save arguments
    args_df = ut.save_arguments(args,now)
################################################# END SECTION OF LOADING PARAMETERS #################################################

################################################# START SECTION OF SINGLE CELL DATA REPROCESSING #################################################
    # Load data and preprocessing
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    logging.info(device)
    torch.cuda.set_device(device)

    # Construct datasets and data loaders
    Xtarget_trainTensor = torch.FloatTensor(Xtarget_train).to(device)
    Xtarget_validTensor = torch.FloatTensor(Xtarget_valid).to(device)

    # Use leiden label if CVAE is applied 
    Ctarget_trainTensor = torch.LongTensor(Ctarget_train).to(device)
    Ctarget_validTensor = torch.LongTensor(Ctarget_valid).to(device)
    
    X_allTensor = torch.FloatTensor(data).to(device)
    C_allTensor = torch.LongTensor(data_c).to(device)


    train_dataset = TensorDataset(Xtarget_trainTensor, Ctarget_trainTensor)
    valid_dataset = TensorDataset(Xtarget_validTensor, Ctarget_validTensor)

    Xtarget_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    Xtarget_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    dataloaders_pretrain = {'train':Xtarget_trainDataLoader,'val':Xtarget_validDataLoader}
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
################################################# END SECTION OF LOADING BULK DATA  #################################################

################################################# START SECTION OF MODEL CUNSTRUCTION  #################################################
    # Construct target encoder
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



    dim_model_out = 2
    # Load AE model
    if reduce_model == "AE":

        source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze)
        source_model.load_state_dict(torch.load(source_model_path))
        source_encoder = source_model
    # Load VAE model
    elif reduce_model in ["VAE"]:
        source_model = PretrainedVAEPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze,z_reparam=bool(args.VAErepram))
        source_model.load_state_dict(torch.load(source_model_path))
        source_encoder = source_model
    logging.info("Load pretrained source model from: "+source_model_path)
           
    source_encoder.to(device)
################################################# END SECTION OF MODEL CUNSTRUCTION  #################################################

################################################# START SECTION OF SC MODEL PRETRAININIG  #################################################
    # Pretrain target encoder
    # Pretain using autoencoder is pretrain is not False
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
                encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                            optimizer=optimizer_e,loss_function=loss_function_e,
                                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)
            elif reduce_model == "VAE":
                encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                optimizer=optimizer_e,
                                n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)
            
            logging.info("Pretrained finished")

        # Before Transfer learning, we test the performance of using no transfer performance:
        # Use vae result to predict 
        embeddings_pretrain = encoder.encode(X_allTensor)

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
    loss_d = nn.CrossEntropyLoss()
    optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
    exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)

    # Set DaNN model
    DaNN_model = DaNN(source_model=source_encoder,target_model=encoder)
    DaNN_model.to(device)

    def loss(x,y,GAMMA=args.mmd_GAMMA):
        result = mmd.mmd_loss(x,y,GAMMA)
        return result

    loss_disrtibution = loss

    # Tran DaNN model
    DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
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
    logging.info("Transfer DaNN finished")
################################################# END SECTION OF TRANSER LEARNING TRAINING #################################################


################################################# START SECTION OF PREPROCESSING FEATURES #################################################
    # Extract feature embeddings 
    # Extract prediction probabilities

    embedding_tensors = encoder.encode(X_allTensor)
    prediction_tensors = source_model.predictor(embedding_tensors)
    embeddings = embedding_tensors.detach().cpu().numpy()
    predictions = prediction_tensors.detach().cpu().numpy()

    # Transform predict8ion probabilities to 0-1 labels

    adata.obs["sens_preds"] = predictions[:,1]
    adata.obs["sens_label"] = predictions.argmax(axis=1)
    adata.obs["sens_label"] = adata.obs["sens_label"].astype('category')
    adata.obs["rest_preds"] = predictions[:,0]

################################################# END SECTION OF PREPROCESSING FEATURES #################################################

################################################# START SECTION OF ANALYSIS AND POST PROCESSING #################################################
################################################# END SECTION OF ANALYSIS AND POST PROCESSING #################################################

################################################# START SECTION OF ANALYSIS FOR BULK DATA #################################################
    # Save adata
    adata.write("saved/adata/"+data_name+now+".h5ad")
################################################# END SECTION OF ANALYSIS FOR BULK DATA #################################################

    t1 = time.time()

    logging.info("End at " + str(t1)+", takes :" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--bulk_data', type=str, default='data/GDSC2_expression.csv',help='Path of the bulk RNA-Seq expression profile')
    parser.add_argument('--label', type=str, default='data/GDSC2_label_9drugs_binary.csv',help='Path of the processed bulk RNA-Seq drug screening annotation')
    parser.add_argument('--sc_data', type=str, default="GSE110894",help='Accession id for testing data, only support pre-built data.')
    parser.add_argument('--drug', type=str, default='Cisplatin',help='Name of the selected drug, should be a column name in the input file of --label')
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

    parser.add_argument('--cluster_res', type=float, default=0.3,help="Resolution of Leiden clustering of scRNA-Seq data, default: 0.3")
    parser.add_argument('--mmd_weight', type=float, default=0.25,help="Weight of the MMD loss of the transfer learning, default: 0.25")
    parser.add_argument('--mmd_GAMMA', type=int, default=1000,help="Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000")

    # train
    parser.add_argument('--bulk_model_path','-s', type=str, default='saved/models/predictor_bulk.pkl',help='Path of the trained predictor in the bulk level')
    parser.add_argument('--sc_model_path', '-p',  type=str, default='saved/models/predictor_sc_',help='Path (prefix) of the trained predictor in the single cell level')
    parser.add_argument('--pretrain', type=str, default='saved/models/encoder_sc.pkl',help='Path of the pre-trained encoder in the single-cell level')

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

    # mis
    parser.add_argument('--logging_file', '-l',  type=str, default='saved/logs/transfer_',help='Path of training log')

    #
    args, unknown = parser.parse_known_args()
    run_main(args)
