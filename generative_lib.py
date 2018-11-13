import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

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