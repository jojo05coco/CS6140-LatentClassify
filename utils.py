## Utility functions

import numpy as np
import torch

## L2 norm
def l2(v):
    #return np.sqrt(v.dot(v)
    return np.sqrt(v.T.dot(v))

# get datapoints from the dataset at random;
# get nsamples from each of the classes
# nl is nsamples
# tailored for mnist/fmnist now

def random_data_points(ds, nl, nsamples):

    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size = 1,
                                             shuffle = True)

    samples = {}
    ndiscovered = 0

    for bidx, (data, target) in enumerate(dataloader):

        if ndiscovered >= nsamples * nl:
            # we are done
            break

        c = target.item()

        if samples.get(c) is None:
            samples[c] = []

        if len(samples[c]) > nsamples:
            continue
        else:
            samples[c].append((data, target))
            ndiscovered = ndiscovered + 1

    # convert dict to a list
    return sum(samples.values(), [])

