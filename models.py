import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np

#import scipy.io as sio
from copy import deepcopy
        
        
class AEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(AEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    #nn.ReLU(),
                    nn.Dropout(drop_out))
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    #nn.ReLU(),
                    nn.Dropout(drop_out))
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1])
                                       ,nn.Sigmoid()
                            )
        # self.feature_extractor =nn.Sequential(
        #     self.encoder,
        #     self.bottleneck
        # )            

             
    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        embedding = self.bottleneck(result)

        return embedding

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs):
        embedding = self.encode(input)
        output = self.decode(embedding)
        return  output        

# Model of Predictor
class Predictor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(Predictor, self).__init__()

        modules = []

        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_out))
            )
            #in_channels = h_dim

        self.predictor = nn.Sequential(*modules)
        #self.output = nn.Linear(hidden_dims[-1], output_dim)

        self.output = nn.Sequential(
                            nn.Linear(hidden_dims[-1],
                                       output_dim),
                                       nn.Sigmoid()
                            )            

    def forward(self, input: Tensor, **kwargs):
        embedding = self.predictor(input)
        output = self.output(embedding)
        return  output
        
        
        
    
# Model of Pretrained P
class PretrainedPredictor(AEBase):
    def __init__(self,
                 # Params from AE model
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3,
                 ### Parameters from predictor models
                 pretrained_weights=None,                 
                 hidden_dims_predictor=[256],
                 drop_out_predictor=0.3,
                 output_dim = 1,
                 freezed = False):
        
        # Construct an autoencoder model
        AEBase.__init__(self,input_dim,latent_dim,h_dims,drop_out)
        
        # Load pretrained weights
        if pretrained_weights !=None:
            self.load_state_dict((torch.load(pretrained_weights)))
        
        ## Free parameters until the bottleneck layer
        if freezed == True:
            bottlenect_reached = False
            for p in self.parameters():
                if ((bottlenect_reached == True)&(p.shape.numel()>self.latent_dim)):
                    break
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
                # Stop until the bottleneck layer
                if p.shape.numel() == self.latent_dim:
                    bottlenect_reached = True
        # Only extract encoder
        del self.decoder
        del self.decoder_input
        del self.final_layer

        self.predictor = Predictor(input_dim=self.latent_dim,
                 output_dim=output_dim,
                 h_dims=hidden_dims_predictor,
                 drop_out=drop_out_predictor)

    def forward(self, input, **kwargs):
        embedding = self.encode(input)
        output = self.predictor(embedding)
        return  output
   
    def predict(self, embedding, **kwargs):
        output = self.predictor(embedding)
        return  output 
     

def vae_loss(recon_x, x, mu, logvar,reconstruction_function,weight=1):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD * weight

class VAEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(VAEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
    
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)
        
        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1],
                            nn.Sigmoid())
                            ) 
        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )
    
    def encode_(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def encode(self, input: Tensor,repram=False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu, log_var = self.encode_(input)

        if (repram==True):
            z = self.reparameterize(mu, log_var)
            return z
        else:
            return mu

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        mu, log_var = self.encode_(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
         M_N = self.params['batch_size']/ self.num_train_imgs,
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] 
        # Account for the minibatch samples from the dataset
        # M_N = self.params['batch_size']/ self.num_train_imgs,
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

class CVAEBase(VAEBase):

    def __init__(self,
                 input_dim,
                 n_conditions,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):

        super(VAEBase, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_condition = n_conditions

        # There are conditions therefore input size is different
        self.encoder_dim = input_dim + n_conditions

        modules_e = []
    
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,self.encoder_dim)
        
        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules_e.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules_e)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules_d = []

        # There are conditions therefore input size is different
        self.decoder_input = nn.Linear(latent_dim+n_conditions, hidden_dims[-1])

        # Replace the output shape
        hidden_dims.reverse()
        hidden_dims[-1]=self.input_dim

        for i in range(len(hidden_dims) - 2):
            modules_d.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules_d)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1],
                            nn.Sigmoid())
                            ) 
        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )
    

    
    def forward(self, input: Tensor,c: Tensor, **kwargs):
        mu, log_var = self.encode_(input,c)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z,c), input, mu, log_var]

    def encode_(self, input: Tensor,c:Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # One hot encoding of inputs 
        c = idx2onehot(c, n=self.n_condition)
        input_c = torch.cat((input, c), dim=-1)

        result = self.encoder(input_c)
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def encode(self, input: Tensor,c:Tensor,repram=False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # One hot encoding of inputs 
        #c = idx2onehot(c, n=self.n_condition)
        #input_c = torch.cat((input, c), dim=-1)

        mu, log_var = self.encode_(input,c)

        if (repram==True):
            z = self.reparameterize(mu, log_var)
            return z
        else:
            return mu

    def decode(self, z: Tensor,c:Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        
        # One hot encoding of inputs 
        c = idx2onehot(c, n=self.n_condition)
        z_c = torch.cat((z, c), dim=-1)

        result = self.decoder_input(z_c)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class PretrainedVAEPredictor(VAEBase):
    def __init__(self,
                 # Params from AE model
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3,
                 ### Parameters from predictor models
                 pretrained_weights=None,                 
                 hidden_dims_predictor=[256],
                 drop_out_predictor=0.3,
                 output_dim = 1,
                 freezed = False,
                 z_reparam=True):
        
        self.z_reparam=z_reparam
        # Construct an autoencoder model
        VAEBase.__init__(self,input_dim,latent_dim,h_dims,drop_out)
        
        # Load pretrained weights
        if pretrained_weights !=None:
            self.load_state_dict((torch.load(pretrained_weights)))
        
        ## Free parameters until the bottleneck layer
        if freezed == True:
            bottlenect_reached = False
            for p in self.parameters():
                if ((bottlenect_reached == True)&(p.shape[0]>self.latent_dim)):
                    break
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
                # Stop until the bottleneck layer
                if p.shape[0] == self.latent_dim:
                    bottlenect_reached = True

        # Only extract encoder
        del self.decoder
        del self.decoder_input
        del self.final_layer

        self.predictor = Predictor(input_dim=self.latent_dim,
                 output_dim=output_dim,
                 h_dims=hidden_dims_predictor,
                 drop_out=drop_out_predictor)

        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )

    def forward(self, input, **kwargs):
        embedding = self.encode(input,repram=self.z_reparam)
        output = self.predictor(embedding)
        return  output

    def predict(self, embedding, **kwargs):
        output = self.predictor(embedding)
        return  output

class DaNN(nn.Module):
    def __init__(self, source_model,target_model,fix_source=False):
        super(DaNN, self).__init__()
        self.source_model = source_model
        if fix_source == True:
            for p in self.parameters():
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
                # Stop until the bottleneck layer
        self.target_model = target_model

    def forward(self, X_source, X_target,C_target=None):
     
        x_src_mmd = self.source_model.encode(X_source)

        if(type(C_target)==type(None)):
            x_tar_mmd = self.target_model.encode(X_target)
        else:
            x_tar_mmd = self.target_model.encode(X_target,C_target)

        y_src = self.source_model.predictor(x_src_mmd)
        return y_src, x_src_mmd, x_tar_mmd
    

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

def g_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD