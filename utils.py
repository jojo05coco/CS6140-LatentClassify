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

def random_data_points(model, ds, nl, nsamples):

    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size = 1,
                                             shuffle = True)

    samples = {}
    ndiscovered = 0

    for bidx, (data, target) in enumerate(dataloader):

        data = data.reshape((1, model.D))

        if ndiscovered >= nsamples * nl:
            # we are done
            break

        c = target.item()

        classified_as = model.test(data).item()

        if (c != classified_as):
            continue

        if samples.get(c) is None:
            samples[c] = []

        if len(samples[c]) > nsamples:
            continue
        else:
            samples[c].append((data, target))
            ndiscovered = ndiscovered + 1

    # convert dict to a list
    return sum(samples.values(), [])

# Given a list of adversarial example; figure out the average L2 norm
# of the adversarial difference
def Adversarial_L2Diff_Mean(examples):

    # filter examples to not have None objects
    examples = filter(lambda ex: ex is not None, examples)

    # extract all L2Norm differences
    l2_diffs = map(lambda ex: ex.adv_diff_norm, examples)

    # return mean of l2_diffs
    return np.mean(l2_diffs)

# check correct classification
def adversarial_correct_classification(exs):

    exs = filter(lambda ex: ex is not None, exs)

    true_class = map(lambda ex: ex.y.item(), exs)
    classified_as = map(lambda ex: np.argmax(ex.x_probs), exs)

    correct = map(lambda tc, cas: tc == cas, true_class, classified_as)

    print "All correct ", all(correct), " : ", len(filter(lambda b: b == False, correct)), len(exs)

    return all(correct)
