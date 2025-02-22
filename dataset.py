##
# All dataset related functions
#

import torch
import numpy as np
import pandas as pd
from torch.utils import data as D

## My own dataset class ########################################################

class CLF_DS(D.Dataset):

    ## constructor
    def __init__(self, train_x_pth, train_y_pth, desc = "dataset"):

        if train_x_pth.endswith("npy"):
            self.X = np.load(train_x_pth)
        else:
            self.X = torch.load(train_x_pth)

        if train_y_pth.endswith("npy"):
            self.Y = np.load(train_y_pth)
        else:
            self.Y = torch.load(train_y_pth)

        #self.X = self.X[0:2000]
        #self.Y = self.Y[0:2000]

        self.desc = desc

        # dataset number of features
        self.D = self.X.shape[1]
        # dataset number of samples
        self.N = self.X.shape[0]
        # number of labels
        if train_y_pth.endswith("pth"):
            self.L = len(set(self.Y.numpy()))
        else:
            self.L = len(set(self.Y))

        # assert
        assert self.Y.shape[0] == self.N

    ## override len
    def __len__(self):
        return self.N

    ## override getitem
    def __getitem__(self, idx):
        #x = self.dt[idx][:self.latent_space_dim]
        #y = self.dt[idx][self.latent_space_dim:].type(torch.long)
        #return x, y

        x = self.X[idx]
        y = self.Y[idx]
        return x, y

    def __repr__(self):

        s = ""
        s = s + " Dataset description - " + str(self.desc) + "\n"
        s = s + " N samples - " + str(self.N) + "\n"
        s = s + " NFeatures - " + str(self.D) + "\n"

        return s
