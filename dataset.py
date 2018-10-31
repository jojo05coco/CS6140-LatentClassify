##
# All dataset related functions
#

import torch
import pandas as pd
from torch.utils import data as D

## My own dataset class ########################################################

class DS(D.Dataset):

    ## constructor
    def __init__(self, csvfname, latent_space_dim):

        self.dsname = "simple dataset"
        self.df = pd.read_csv(csvfname)
        self.dt = torch.tensor(self.df.values, dtype = torch.float)
        self.latent_space_dim = latent_space_dim

    ## override len
    def __len__(self):
        return len(self.df)

    ## override getitem
    def __getitem__(self, idx):
        x = self.dt[idx][:self.latent_space_dim]
        y = self.dt[idx][self.latent_space_dim:].type(torch.long)
        return x, y
