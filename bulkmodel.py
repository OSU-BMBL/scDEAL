import argparse
import logging
import sys
import time
import warnings
import os
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
from models import (AEBase,PretrainedPredictor, PretrainedVAEPredictor, VAEBase)
import matplotlib
import random
seed=42
torch.manual_seed(seed)
#np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#from transformers import *
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

def run_main(args):

    # Extract parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    select_drug = args.drug
    na = args.missing_value
    data_path = args.data
    label_path = args.label
    test_size = args.test_size
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    log_path = args.log
    batch_size = args.batch_size
    encoder_hdims = args.encoder_h_dims.split(",")
    preditor_hdims = args.predictor_h_dims.split(",")
    reduce_model = args.dimreduce
    sampling = args.sampling
    PCA_dim = args.PCA_dim

    encoder_hdims = list(map(int, encoder_hdims) )
    preditor_hdims = list(map(int, preditor_hdims) )
    load_model = bool(args.load_source_model)

    

    para = str(args.bulk)+"_data_"+str(args.data_name)+"_drug_"+str(args.drug)+"_bottle_"+str(args.bottleneck)+"_edim_"+str(args.encoder_h_dims)+"_pdim_"+str(args.predictor_h_dims)+"_model_"+reduce_model+"_dropout_"+str(args.dropout)+"_gene_"+str(args.printgene)+"_lr_"+str(args.lr)+"_mod_"+str(args.mod)+"_sam_"+str(args.sampling)    #(para)
    now=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    #print(preditor_path )
    #model_path = args.bulk_model + para 
    preditor_path = args.bulk_model + para 
    bulk_encoder = args.bulk_encoder+para
    # Read data
    data_r=pd.read_csv(data_path,index_col=0)
    label_r=pd.read_csv(label_path,index_col=0)
    if args.bulk == 'old':
        data_r=data_r[0:805]
        label_r=label_r[0:805]
    elif args.bulk == 'new':
        data_r=data_r[805:data_r.shape[0]]
        label_r=label_r[805:label_r.shape[0]]        
    else:
        print("two databases combine")
    label_r=label_r.fillna(na)
    ut.save_arguments(args,now)


    # Initialize logging and std out
    out_path = log_path+now+".err"
    log_path = log_path+now+".log"

    out=open(out_path,"w")
    sys.stderr=out
    

    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.getLogger('matplotlib.font_manager').disabled = True

    logging.info(args)

    # Filter out na values
    selected_idx = label_r.loc[:,select_drug]!=na

    if(g_disperson!=None):
        hvg,adata = ut.highly_variable_genes(data_r,min_disp=g_disperson)
        # Rename columns if duplication exist
        data_r.columns = adata.var_names
        # Extract hvgs
        data = data_r.loc[selected_idx.index,hvg]
    else:
        data = data_r.loc[selected_idx.index,:]

    # Do PCA if PCA_dim!=0
    if PCA_dim !=0 :
        data = PCA(n_components = PCA_dim).fit_transform(data)
    else:
        data = data
        
    # Extract labels
    label = label_r.loc[selected_idx.index,select_drug]
    data_r = data_r.loc[selected_idx.index,:]

    # Scaling data
    mmscaler = preprocessing.MinMaxScaler()

    data = mmscaler.fit_transform(data)
    label = label.values.reshape(-1,1)


    le = LabelEncoder()
    label = le.fit_transform(label)
    dim_model_out = 2

    #label = label.values.reshape(-1,1)

    logging.info(np.std(data))
    logging.info(np.mean(data))

    # Split traning valid test set
    X_train_all, X_test, Y_train_all, Y_test = train_test_split(data, label, test_size=test_size, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=valid_size, random_state=42)
    # sampling method
    if sampling == "no":
        X_train,Y_train=sam.nosampling(X_train,Y_train)
        logging.info("nosampling")
    elif sampling =="upsampling":
        X_train,Y_train=sam.upsampling(X_train,Y_train)
        logging.info("upsampling")
    elif sampling =="downsampling":
        X_train,Y_train=sam.downsampling(X_train,Y_train)
        logging.info("downsampling")
    elif  sampling=="SMOTE":
        X_train,Y_train=sam.SMOTEsampling(X_train,Y_train)
        logging.info("SMOTE")
    else:
        logging.info("not a legal sampling method")

    # Select the Training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    #print(device)
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    #logging.info(device)
    torch.cuda.set_device(device)
    print(device)
    # Construct datasets and data loaders
    X_trainTensor = torch.FloatTensor(X_train).to(device)
    X_validTensor = torch.FloatTensor(X_valid).to(device)
    X_testTensor = torch.FloatTensor(X_test).to(device)

    Y_trainTensor = torch.LongTensor(Y_train).to(device)
    Y_validTensor = torch.LongTensor(Y_valid).to(device)

    # Preprocess data to tensor
    train_dataset = TensorDataset(X_trainTensor, X_trainTensor)
    valid_dataset = TensorDataset(X_validTensor, X_validTensor)

    X_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    X_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    # construct TensorDataset
    trainreducedDataset = TensorDataset(X_trainTensor, Y_trainTensor)
    validreducedDataset = TensorDataset(X_validTensor, Y_validTensor)

    trainDataLoader_p = DataLoader(dataset=trainreducedDataset, batch_size=batch_size, shuffle=True)
    validDataLoader_p = DataLoader(dataset=validreducedDataset, batch_size=batch_size, shuffle=True)
    bulk_X_allTensor = torch.FloatTensor(data).to(device)
    bulk_Y_allTensor = torch.LongTensor(label).to(device)
    dataloaders_train = {'train':trainDataLoader_p,'val':validDataLoader_p}
    print("bulk_X_allRensor",bulk_X_allTensor.shape)
    if(bool(args.pretrain)!=False):
        dataloaders_pretrain = {'train':X_trainDataLoader,'val':X_validDataLoader}
        if reduce_model == "VAE":
            encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        if reduce_model == 'AE':
            encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        if reduce_model =='DAE':            
            encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        
        #if torch.cuda.is_available():
        #    encoder.cuda()

        #logging.info(encoder)
        encoder.to(device)
        #print(encoder)
        optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

        if reduce_model == "AE":
            encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                        optimizer=optimizer_e,loss_function=loss_function_e,
                                        n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder)
        elif reduce_model == "VAE":
            encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                            optimizer=optimizer_e,
                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder)
        if reduce_model == "DAE":
            encoder,loss_report_en = t.train_DAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                        optimizer=optimizer_e,loss_function=loss_function_e,
                                        n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder)                          
                                    
        
        #logging.info("Pretrained finished")

    # Defined the model of predictor 
    if reduce_model == "AE":
        model = PretrainedPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                                hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                pretrained_weights=bulk_encoder,freezed=bool(args.freeze_pretrain),drop_out=args.dropout,drop_out_predictor=args.dropout)
    if reduce_model == "DAE":
        model = PretrainedPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                                hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                pretrained_weights=bulk_encoder,freezed=bool(args.freeze_pretrain),drop_out=args.dropout,drop_out_predictor=args.dropout)                                
    elif reduce_model == "VAE":
        model = PretrainedVAEPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                        hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                        pretrained_weights=bulk_encoder,freezed=bool(args.freeze_pretrain),z_reparam=bool(args.VAErepram),drop_out=args.dropout,drop_out_predictor=args.dropout)
    #print("@@@@@@@@@@@")
    logging.info("Current model is:")
    logging.info(model)
    #if torch.cuda.is_available():
    #    model.cuda()
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)


    loss_function = nn.CrossEntropyLoss()

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    # Train prediction model
    #print("1111")
    model,report = t.train_predictor_model(model,dataloaders_train,
                                            optimizer,loss_function,epochs,exp_lr_scheduler,load=load_model,save_path=preditor_path)
    if (args.printgene=='T'):
        import scanpypip.preprocessing as pp
        bulk_adata = pp.read_sc_file(data_path)
        #print('pp')
        ## bulk test predict critical gene
        import scanpy as sc
        #import scanpypip.utils as uti
        from captum.attr import IntegratedGradients
        #bulk_adata = bulk_adata
        #print(bulk_adata) 
        bulk_pre = model(bulk_X_allTensor).detach().cpu().numpy()  
        bulk_pre = bulk_pre.argmax(axis=1)
        #print(model)
        #print(bulk_pre.shape)
        # Caculate integrated gradient
        ig = IntegratedGradients(model)
        
        df_results_p = {}
        target=1
        attr, delta =  ig.attribute(bulk_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        
        #attr, delta =  ig.attribute(bulk_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()
        
        np.savetxt("ori_result/"+args.data_name+"bulk_gradient.txt",attr,delimiter = " ")
        from pandas.core.frame import DataFrame
        DataFrame(bulk_pre).to_csv("ori_result/"+args.data_name+"bulk_lab.csv")
    dl_result = model(X_testTensor).detach().cpu().numpy()


    lb_results = np.argmax(dl_result,axis=1)
    #pb_results = np.max(dl_result,axis=1)
    pb_results = dl_result[:,1]

    report_dict = classification_report(Y_test, lb_results, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    ap_score = average_precision_score(Y_test, pb_results)
    auroc_score = roc_auc_score(Y_test, pb_results)

    report_df['auroc_score'] = auroc_score
    report_df['ap_score'] = ap_score

    report_df.to_csv("save/logs/" + reduce_model + select_drug+now + '_report.csv')

    #logging.info(classification_report(Y_test, lb_results))
    #logging.info(average_precision_score(Y_test, pb_results))
    #logging.info(roc_auc_score(Y_test, pb_results))

    model = DummyClassifier(strategy='stratified')
    model.fit(X_train, Y_train)
    yhat = model.predict_proba(X_test)
    naive_probs = yhat[:, 1]

    ut.plot_roc_curve(Y_test, naive_probs, pb_results, title=str(roc_auc_score(Y_test, pb_results)),
                        path="save/figures/" + reduce_model + select_drug+now + '_roc.pdf')
    ut.plot_pr_curve(Y_test,pb_results,  title=average_precision_score(Y_test, pb_results),
                    path="save/figures/" + reduce_model + select_drug+now + '_prc.pdf')
    print("bulk_model finished")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--data', type=str, default='data/ALL_expression.csv',help='Path of the bulk RNA-Seq expression profile')
    parser.add_argument('--label', type=str, default='data/ALL_label_binary_wf.csv',help='Path of the processed bulk RNA-Seq drug screening annotation')
    parser.add_argument('--result', type=str, default='save/results/result_',help='Path of the training result report files')
    parser.add_argument('--drug', type=str, default='I-BET-762',help='Name of the selected drug, should be a column name in the input file of --label')
    parser.add_argument('--missing_value', type=int, default=1,help='The value filled in the missing entry in the drug screening annotation, default: 1')
    parser.add_argument('--test_size', type=float, default=0.2,help='Size of the test set for the bulk model traning, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,help='Size of the validation set for the bulk model traning, default: 0.2')
    parser.add_argument('--var_genes_disp', type=float, default=None,help='Dispersion of highly variable genes selection when pre-processing the data. \
                         If None, all genes will be selected .default: None')
    parser.add_argument('--sampling', type=str, default=None,help='Samping method of training data for the bulk model traning. \
                        Can be upsampling, downsampling, or SMOTE. default: None')
    parser.add_argument('--PCA_dim', type=int, default=0,help='Number of components of PCA  reduction before training. If 0, no PCA will be performed. Default: 0')

    # trainv
    parser.add_argument('--bulk_encoder','-e', type=str, default='save/bulk_encoder/',help='Path of the pre-trained encoder in the bulk level')
    parser.add_argument('--pretrain', type=int, default=1,help='Whether to perform pre-training of the encoder. 0: do not pretraing, 1: pretrain. Default: 0')
    parser.add_argument('--lr', type=float, default=1e-2,help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=500,help='Number of epoches training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=200,help='Number of batch size when training. Default: 200')
    parser.add_argument('--bottleneck', type=int, default=32,help='Size of the bottleneck layer of the model. Default: 32')
    parser.add_argument('--dimreduce', type=str, default="AE",help='Encoder model type. Can be AE or VAE. Default: AE')
    parser.add_argument('--freeze_pretrain', type=int, default=0,help='Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0')
    parser.add_argument('--encoder_h_dims', type=str, default="512,256",help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--predictor_h_dims', type=str, default="16,8",help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 16,8')
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--data_name', type=str, default="GSE110894",help='Accession id for testing data, only support pre-built data.')
    # misc
    parser.add_argument('--bulk_model', '-p',  type=str, default='save/bulk_pre/',help='Path of the trained prediction model in the bulk level')
    parser.add_argument('--log', '-l',  type=str, default='save/logs/log',help='Path of training log')
    parser.add_argument('--load_source_model',  type=int, default=0,help='Load a trained bulk level or not. 0: do not load, 1: load. Default: 0')
    parser.add_argument('--mod', type=str, default='new',help='Embed the cell type label to regularized the training: new: add cell type info, ori: do not add cell type info. Default: new')
    parser.add_argument('--printgene', type=str, default='T',help='Print the cirtical gene list: T: print. Default: T')
    parser.add_argument('--dropout', type=float, default=0.3,help='Dropout of neural network. Default: 0.3')
    parser.add_argument('--bulk', type=str, default='integrate',help='Selection of the bulk database.integrate:both dataset. old: GDSC. new: CCLE. Default: integrate')
    warnings.filterwarnings("ignore")

    args, unknown = parser.parse_known_args()
    matplotlib.use('Agg')

    run_main(args)

