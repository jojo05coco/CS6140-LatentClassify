import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

class AE_1L(torch.nn.Module):
    
    def __init__(self, num_features, num_hidden_1):
        super(AE_1L, self).__init__()

        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
        

    def encoder(self, x):
        encoded = self.linear_1(x)
        encoded = F.leaky_relu(encoded)
        return encoded
    
    def decoder(self, x):
        logits = self.linear_2(x)
        decoded = F.sigmoid(logits)
        return decoded    
        
    def forward(self, x):
        encoded = self.encoder(x);
        decoded = self.decoder(encoded);
        return decoded

    
class AE_1L_gen(torch.nn.Module):
    
    def __init__(self, num_features, num_hidden_1):
        super(AE_1L_gen, self).__init__()

        self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
        
    def forward(self, x):
        x = self.linear_2(x)
        decoded = F.sigmoid(x)
        return decoded
    

class VAE(torch.nn.Module):

    def __init__(self, num_features, num_hidden_1, num_latent):
        super(VAE, self).__init__()
        
        ### ENCODER
        self.hidden_1 = torch.nn.Linear(num_features, num_hidden_1)
        self.z_mean = torch.nn.Linear(num_hidden_1, num_latent)
        self.z_log_var = torch.nn.Linear(num_hidden_1, num_latent)
        
        
        ### DECODER
        self.linear_3 = torch.nn.Linear(num_latent, num_hidden_1)
        self.linear_4 = torch.nn.Linear(num_hidden_1, num_features)

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def encoder(self,x):
        x = self.hidden_1(x)
        x = F.relu(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return [encoded,z_mean,z_log_var]
    
    def decoder(self,x):
        x = self.linear_3(x)
        x = F.relu(x)
        x = self.linear_4(x)
        decoded = F.sigmoid(x)
        return decoded
        
    def forward(self, x):
        [encoded,z_mean,z_log_var] = self.encoder(x)
        decoded                    = self.decoder(encoded)
        return z_mean, z_log_var, encoded, decoded
    
    
class VAE_gen(torch.nn.Module):
    
    def __init__(self, num_features, num_hidden_1, num_latent):
        super(VAE_gen, self).__init__()

        self.linear_3 = torch.nn.Linear(num_latent, num_hidden_1)
        self.linear_4 = torch.nn.Linear(num_hidden_1, num_features)
        
    def forward(self,x):
        x = self.linear_3(x)
        x = F.relu(x)
        x = self.linear_4(x)
        decoded = F.sigmoid(x)
        return decoded
    
# freeze weigths
def froze_weights(net):
    for param in net.parameters():
        param.requires_grad = False
        
########## Projection ###############
def L2_Project(generator, Gstar, maxit, x0, gamma):
    
    xk = Variable(x0, requires_grad=True)
    
    for i in range(maxit):
        Gxk = generator(xk);
        l2_loss_var = 0.5*((Gxk-Gstar)**2).sum()
        l2_loss_var.backward()
        xk.data -= gamma*xk.grad.data
        xk.grad.data.zero_();  
    
    return [l2_loss_var,xk]


def L1_Project(generator, Gstar, maxit, x0, gamma):
    
    xk = Variable(x0, requires_grad=True)
    
    for i in range(maxit):
        Gxk = generator(xk);
        l1_loss_var = torch.abs(Gxk-Gstar).sum()
        l1_loss_var.backward()
        xk.data -= gamma*xk.grad.data
        xk.grad.data.zero_();  
    
    return [l1_loss_var,xk]


def Full_Projection(generator, Gstar, maxit, gamma, rand_init, sigma, x0, Project):
    if(rand_init == 0):
        [loss_var, xk] = Project(generator, Gstar, maxit, x0, gamma)
        return [loss_var,xk]
    else:
        min_loss = 10**6
        x_min    = 0*x0;
        for j in range(rand_init):
            xk = sigma*torch.randn(x0.size());
            [l2_loss_var, xk] = Project(generator, Gstar, maxit, xk, gamma)
            if(min_loss > l2_loss_var):
                min_loss = l2_loss_var
                x_min    = xk
        return [min_loss,x_min]