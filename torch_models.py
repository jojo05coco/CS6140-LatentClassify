import torch

from dataset import DS

## Warning ignore
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os

## Globals
LR = 0.0001 # Learning Rate
BS = 3      # Batch size for training
# Refer to ./test/train_dataset.txt or ./test/test_dataset.txt for the format of input files
TRAIN_DATASET_PATH =  "./test/train_dataset.txt"
TEST_DATASET_PATH  = "./test/test_dataset.txt"

LATENT_SPACE_DIM = 2
NCLASSES         = 2

NEPOCH = 1000

## global helpers ##

def train(train_loader, model, optimizer, epoch):
    model.train()

    avg_loss = 0.0
    count = 0.0

    for bidx, (data, targets) in enumerate(train_loader):

        #print "start --------------------------------------------------------------"
        #for param in model.parameters():
        #    print param.grad
        #for param in model.parameters():
        #    print param
        #print "--------------------------------------------------------------------"

        # make 2D target tensor; 1D
        targets = torch.tensor(targets.numpy().flatten())

        # setup
        optimizer.zero_grad()

        # forward
        log_softmax = model(data)

        # prediction
        loss = torch.nn.functional.nll_loss(log_softmax, targets)

        # back prop
        loss.backward()

        avg_loss = avg_loss + loss.item()
        count = count + 1

        #for param in model.parameters():
        #    print param.grad

        # update weights
        optimizer.step()

        #print "end -----------------------------------------------------------------"
        #for param in model.parameters():
        #    print param.grad
        #for param in model.parameters():
        #    print param
        #print "--------------------------------------------------------------------"

    if epoch % 100 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 1 * len(data), len(train_loader.dataset),
                100. * 1 / len(train_loader), avg_loss / count))
        c = raw_input("enter to cont..")

def test(test_loader, model):

    model.eval()
    test_loss = 0.0
    correct   = 0
    total     = 0

    with torch.no_grad():

        for data, targets in test_loader:

            # make 2D target tensor; 1D
            targets = torch.tensor(targets.numpy().flatten())

            log_softmax = model(data)

            loss = torch.nn.functional.nll_loss(log_softmax, targets)

            test_loss = test_loss + loss.item()

            pred_targets = log_softmax[0].max(0)[1]
            if targets.item() == pred_targets.item():
                correct = correct + 1

            total = total + 1

    # how does this work ??
    print "Test loss - ", test_loss, " Accuracy - ", correct, " / ", total

## read the data

# train and test dataset
train_ds = DS(TRAIN_DATASET_PATH, LATENT_SPACE_DIM)
test_ds  = DS(TEST_DATASET_PATH, LATENT_SPACE_DIM)

# train and test dataset loader
train_loader = torch.utils.data.DataLoader(train_ds, batch_size = BS, shuffle = True)
test_loader  = torch.utils.data.DataLoader(test_ds)

## Logistic regression #########################################################

# model
logistic_model = torch.nn.Sequential(torch.nn.Linear(LATENT_SPACE_DIM, NCLASSES),
                                     torch.nn.LogSoftmax(1))
# optimizer
logistic_optimizer = torch.optim.SGD(logistic_model.parameters(), lr = LR)

print "Training logistic regression ..."
for i in range(1, NEPOCH):
    train(train_loader, logistic_model, logistic_optimizer, i)

    if i % 100 == 0:
        test(test_loader, logistic_model)

## Shallow network; 2 linear + softmax #########################################

HIDDEN_LAYER_DIM = LATENT_SPACE_DIM + 4 # arbitrary; to be configured

# model
shallow_model = torch.nn.Sequential(torch.nn.Linear(LATENT_SPACE_DIM, HIDDEN_LAYER_DIM),
                                    torch.nn.Linear(HIDDEN_LAYER_DIM, NCLASSES),
                                    torch.nn.LogSoftmax(1))
# optimizer
shallow_optimizer = torch.optim.SGD(shallow_model.parameters(), lr = LR)

print "Training shallow network"
for i in range(1, NEPOCH):
    train(train_loader, shallow_model, shallow_optimizer, i)

    if i % 100 == 0:
        test(test_loader, shallow_model)
