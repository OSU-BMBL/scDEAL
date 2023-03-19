import copy
import logging
import os

import numpy as np

import torch
from torch import nn
from tqdm import tqdm

from models import vae_loss
from sklearn.metrics.pairwise import cosine_similarity
### loss2
import copy

from scipy.spatial import distance_matrix, minkowski_distance, distance
import networkx as nx
from igraph import *



def train_AE_model(net,data_loaders={},optimizer=None,loss_function=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf

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

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                #print(x)
                output = net(x)
                # compute loss
                loss = loss_function(output, x)      

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
  
            epoch_loss = running_loss / n_iters
            #print(epoch_loss)
            
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts
    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)           
    
    return net, loss_train
    
def train_DAE_model(net,data_loaders={},optimizer=None,loss_function=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf

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

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):
     
                z = x
                y = np.random.binomial(1, 0.2, (z.shape[0], z.shape[1]))
                z[np.array(y, dtype= bool),] = 0
                x.requires_grad_(True)
                # encode and decode 
                output = net(z)
                # compute loss
                loss = loss_function(output, x)      

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
  
            epoch_loss = running_loss / n_iters

            print(epoch_loss)
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts
    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)           
    
    return net, loss_train    

def train_VAE_model(net,data_loaders={},optimizer=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl",best_model_cache = "drive"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    if best_model_cache == "memory":
        best_model_wts = copy.deepcopy(net.state_dict())
    else:
        torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    best_loss = np.inf

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

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                output = net(x)
                # compute loss

                #losses = net.loss_function(*output, M_N=data_loaders[phase].batch_size/dataset_sizes[phase])      
                #loss = losses["loss"]

                recon_loss = nn.MSELoss(reduction="sum")

                loss = vae_loss(output[0],output[1],output[2],output[3],recon_loss,data_loaders[phase].batch_size/dataset_sizes[phase])

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
                
            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_loss = running_loss / n_iters

            
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    return net, loss_train

def train_CVAE_model(net,data_loaders={},optimizer=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl",best_model_cache = "drive"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    if best_model_cache == "memory":
        best_model_wts = copy.deepcopy(net.state_dict())
    else:
        torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    best_loss = np.inf

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

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, c) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                output = net(x,c)
                # compute loss

                #losses = net.loss_function(*output, M_N=data_loaders[phase].batch_size/dataset_sizes[phase])      
                #loss = losses["loss"]

                recon_loss = nn.MSELoss(reduction="sum")

                loss = vae_loss(output[0],output[1],output[2],output[3],recon_loss,data_loaders[phase].batch_size/dataset_sizes[phase])

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
                
            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_loss = running_loss / n_iters

            
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    return net, loss_train

def train_predictor_model(net,data_loaders,optimizer,loss_function,n_epochs,scheduler,load=False,save_path="model.pkl"):

    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf



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

            # N iter s calculated
            n_iters = len(data_loaders[phase])

            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, y) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                output = net(x)
                # compute loss
                loss = loss_function(output, y)      

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            

            epoch_loss = running_loss / n_iters
            print(epoch_loss)
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts
        torch.save(best_model_wts, save_path)
        
    net.load_state_dict(best_model_wts)           
    
    return net, loss_train

def train_ADDA_model(
    source_encoder, target_encoder, discriminator,
    source_loader, target_loader,
    dis_loss, target_loss,
    optimizer, d_optimizer,
    scheduler,d_scheduler,
    n_epochs,device,save_path="saved/models/model.pkl",
    args=None):
    
    
    target_dataset_sizes = {x: target_loader[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    source_dataset_sizes = {x: source_loader[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}

    dataset_sizes = {x: min(target_dataset_sizes[x],source_dataset_sizes[x]) for x in ['train', 'val']}

    loss_train = {}
    loss_d_train = {}

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                source_encoder.eval()
                target_encoder.train()  # Set model to training mode
                discriminator.train()  # Set model to training mode

            else:
                source_encoder.eval()
                target_encoder.eval()  # Set model to evaluate mode
                discriminator.eval()  # Set model to training mode

            running_loss = 0.0
            d_running_loss = 0.0

                #losses, d_losses = AverageMeter(), AverageMeter()
            n_iters = min(len(source_loader[phase]), len(target_loader[phase]))
            source_iter, target_iter = iter(source_loader[phase]), iter(target_loader[phase])

            # Iterate over data.
            # for data in data_loaders[phase]:
            for iter_i in range(n_iters):
                source_data, source_target = source_iter.next()
                target_data, target_target = target_iter.next()
                source_data = source_data.to(device)
                target_data = target_data.to(device)
                s_bs = source_data.size(0)
                t_bs = target_data.size(0)

                D_input_source = source_encoder.encode(source_data)
                D_input_target = target_encoder.encode(target_data)
                D_target_source = torch.tensor(
                    [0] * s_bs, dtype=torch.long).to(device)
                D_target_target = torch.tensor(
                    [1] * t_bs, dtype=torch.long).to(device)

                # Add adversarial label    
                D_target_adversarial = torch.tensor(
                    [0] * t_bs, dtype=torch.long).to(device)
                
                # train Discriminator
                # Please fix it here to be a classifier
                D_output_source = discriminator(D_input_source)
                D_output_target = discriminator(D_input_target)
                D_output = torch.cat([D_output_source, D_output_target], dim=0)
                D_target = torch.cat([D_target_source, D_target_target], dim=0)
                d_loss = dis_loss(D_output, D_target)


                d_optimizer.zero_grad()

                if phase == 'train':
                    d_loss.backward()
                    d_optimizer.step()
                
                d_running_loss += d_loss.item()

                D_input_target = target_encoder.encode(target_data)
                D_output_target = discriminator(D_input_target)
                loss = dis_loss(D_output_target, D_target_adversarial)

                optimizer.zero_grad()


                if phase == 'train':

                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
            

            epoch_loss = running_loss/n_iters
            d_epoch_loss = d_running_loss/n_iters


            if phase == 'train':
                scheduler.step(epoch_loss)
                d_scheduler.step(d_epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            d_last_lr = d_scheduler.optimizer.param_groups[0]['lr']

            loss_train[epoch,phase] = epoch_loss
            loss_d_train[epoch,phase] = d_epoch_loss

            logging.info('Discriminator {} Loss: {:.8f}. Learning rate = {}'.format(phase, d_epoch_loss,d_last_lr))
            logging.info('Encoder {} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))

            # if phase == 'val' and epoch_loss < best_loss:
            #     best_loss = epoch_loss
            #     best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts
        torch.save(discriminator.state_dict(), save_path+"_d.pkl")
        torch.save(target_encoder.state_dict(), save_path+"_te.pkl")

    #net.load_state_dict(best_model_wts)           
    
    return discriminator,target_encoder, loss_train, loss_d_train

def train_DaNN_model(net,source_loader,target_loader,
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
                loss_c = loss_function(y_pre, y_src)      
                loss_mmd = dist_loss(x_src_mmd, x_tar_mmd)

                loss = loss_c + weight * loss_mmd


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

                # Iterate over batch
                batch_j += 1
                if batch_j >= len(list_tar):
                    batch_j = 0

            # Average epoch loss
            epoch_loss = running_loss / n_iters
            epoch_mmd = running_mmd/n_iters

            # Step schedular
            if phase == 'train':
                scheduler.step(epoch_loss)
            
            # Savle loss
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            mmd_train[epoch,phase] = epoch_mmd

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

    return net, [loss_train,mmd_train]

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
                    load=False,save_path="save/model.pkl",best_model_cache = "drive",top_models=5,k=10,device="cuda"):

    if(load!=False):
        if(os.path.exists(save_path)):
            try:
                net.load_state_dict(torch.load(save_path))           
                return net, 0,0,0
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
                    if(device=="cuda"):
                        loss_s = torch.tensor(loss_s).cuda()
                    else:
                        loss_s = torch.tensor(loss_s).cpu()
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