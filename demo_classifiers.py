## Demo program on how to use the classifiers

import os
import datetime

import numpy as np
import torch

from dataset import CLF_DS

from classifiers import Logistic_1L, Logistic_2L, Logistic_LRL, Logistic_LRLRL

from svm_classifier import SVM_Classifier

## Utils #######################################################################

def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def make_npy(ds_path):

    if ds_path.endswith("npy"):
        return ds_path

    assert ds_path.endswith("pth")

    # npy path
    ds_path_npy = list(ds_path)
    ds_path_npy[-1] = 'y'
    ds_path_npy[-2] = 'p'
    ds_path_npy[-3] = 'n'
    ds_path_npy = "".join(ds_path_npy)
    assert ds_path_npy.endswith("npy")

    # load pth
    ds = torch.load(ds_path)
    # convert to npy
    ds_npy = ds.detach().numpy()

    f = open(ds_path_npy, "wb+")
    np.save(f, ds_npy)
    f.close()

    return ds_path_npy

## Neural networks #############################################################

def train_nn(train_x, train_y, test_x, test_y, ds_desc):

    train_ds = CLF_DS(train_x, train_y, ds_desc)
    test_ds  = CLF_DS(test_x, test_y, ds_desc)

    clf = Logistic_LRL(train_ds, test_ds, batch_size = 1000, epochs = 150,
                       report_folder = ds_desc + "_Logistic_LRL_" + timestamp())
    clf.train()

    clf = Logistic_1L(train_ds, test_ds, batch_size = 1000, epochs = 150,
                      report_folder = ds_desc + "Logistic_1L_" + timestamp())
    clf.train()

    clf = Logistic_2L(train_ds, test_ds, batch_size = 1000, epochs = 150,
                      report_folder = ds_desc + "Logistic_2L_" + timestamp())
    clf.train()

## SVM classifier ##############################################################

#Cs = [1, 10, 20, 50]
#
#for c in Cs:
#    report_folder = "SVM_" + str(c)
#    svm_clf = SVM_Classifier(latent_codes_train,
#                             targets_train,
#                             latent_codes_test,
#                             targets_test,
#                             C = c,
#                             report_folder = report_folder)
#
#    svm_clf.train()
#    svm_clf.report()

## AE MNIST
train_x = make_npy("./test/AE_MNIST-latent_tr.pth")
train_y = make_npy("./test/AE_MNIST-targets_tr.pth")
test_x  = make_npy("./test/AE_MNIST-latent_ts.pth")
test_y  = make_npy("./test/AE_MNIST-targets_ts.pth")
train_nn(train_x, train_y, test_x, test_y, "MNIST_AE")

## VAE MNIST
train_x = make_npy("./test/VAE_MNIST-latent_tr.pth")
train_y = make_npy("./test/VAE_MNIST-targets_tr.pth")
test_x  = make_npy("./test/VAE_MNIST-latent_ts.pth")
test_y  = make_npy("./test/VAE_MNIST-targets_ts.pth")
train_nn(train_x, train_y, test_x, test_y, "MNIST_VAE")

## AE MNIST L2
train_x = make_npy("./test/AE_MNIST_L2latent_tr.npy")
train_y = make_npy("./test/AE_MNIST-targets_tr.pth")
test_x  = make_npy("./test/AE_MNIST_L2latent_ts.npy")
test_y  = make_npy("./test/AE_MNIST-targets_ts.pth")
train_nn(train_x, train_y, test_x, test_y, "MNIST_AE_L2")

## VAE MNIST L2
train_x = make_npy("./test/VAE_MNIST_L2latent_tr.pth")
train_y = make_npy("./test/VAE_MNIST-targets_tr.pth")
test_x  = make_npy("./test/VAE_MNIST_L2latent_ts.pth")
test_y  = make_npy("./test/VAE_MNIST-targets_ts.pth")
train_nn(train_x, train_y, test_x, test_y, "MNIST_VAE_L2")

## AE FMNIST
train_x = make_npy("./test/AE_FMNIST-latent_tr.pth")
train_y = make_npy("./test/AE_FMNIST-targets_tr.pth")
test_x  = make_npy("./test/AE_FMNIST-latent_ts.pth")
test_y  = make_npy("./test/AE_FMNIST-targets_ts.pth")
train_nn(train_x, train_y, test_x, test_y, "FMNIST_AE")

## VAE FMNIST
train_x = make_npy("./test/VAE_FMNIST-latent_tr.pth")
train_y = make_npy("./test/VAE_FMNIST-targets_tr.pth")
test_x  = make_npy("./test/VAE_FMNIST-latent_ts.pth")
test_y  = make_npy("./test/VAE_FMNIST-targets_ts.pth")
train_nn(train_x, train_y, test_x, test_y, "FMNIST_VAE")

## AE FMNIST L2
train_x = make_npy("./test/AE_FMNIST_L2latent_tr.npy")
train_y = make_npy("./test/AE_FMNIST-targets_tr.pth")
test_x  = make_npy("./test/AE_FMNIST_L2latent_ts.pth")
test_y  = make_npy("./test/AE_FMNIST-targets_ts.pth")
train_nn(train_x, train_y, test_x, test_y, "FMNIST_AE_L2")

## VAE FMNIST L2
train_x = make_npy("./test/VAE_FMNIST_L2latent_tr.pth")
train_y = make_npy("./test/VAE_FMNIST-targets_tr.pth")
test_x  = make_npy("./test/VAE_FMNIST_L2latent_ts.pth")
test_y  = make_npy("./test/VAE_FMNIST-targets_ts.pth")
train_nn(train_x, train_y, test_x, test_y, "FMNIST_VAE_L2")
