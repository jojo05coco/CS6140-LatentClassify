
import os
import sys
import datetime

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pickle

import matplotlib.pyplot as plt
import numpy as np

from classifiers import Logistic_LRL, Logistic_2L, Logistic_1L, AdversarialExample
from dataset import CLF_DS

from file_utils import create_folder, delete_folder, folder_exists
from utils import random_data_points

# folder containing/to-contain the dataset
root = "./data"

## Utils #######################################################################

def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def train_mnist():

    trans = transforms.Compose([transforms.ToTensor()])
    train_set = dset.MNIST(root = root, train = True, transform = trans, download = True)
    test_set = dset.MNIST(root = root, train = False, transform = trans, download = True)

    #train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = 1, shuffle = True)

    clf = Logistic_LRL(train_set, test_set, batch_size = 1000, epochs = 150, n_h1 = 30,
                       report_folder = "Logistic_LRL_" + timestamp(), is_CLF_DS = False)
    clf.train()

    clf = Logistic_2L(train_set, test_set, batch_size = 1000, epochs = 50, n_h1 = 30,
                       report_folder = "MNIST_img_Logistic_2L_" + timestamp(), is_CLF_DS = False)

    clf.train()

    clf = Logistic_1L(train_set, test_set, batch_size = 1000, epochs = 50,
                       report_folder = "MNIST_img_Logistic_LRL_" + timestamp(), is_CLF_DS = False)
    clf.train()

def generate_advex(model_paths, report_folder = ""):

    if report_folder != "":
        delete_folder(report_folder)
        create_folder(report_folder)
        assert folder_exists(report_folder)

    for model_path in model_paths:

        adv_examples = []

        # model
        print "Model path to open ", model_path
        mf = open(model_path, "rb")
        model = pickle.load(mf)
        mf.close()

        # loaded model
        print "Loaded model description ", model.desc

        # we need 5 data points from each label
        # we have 10 labels
        data_points = random_data_points(model.test_ds, model.L, 5)
        print "Model L ", model.L, " | ", len(data_points)
        assert len(data_points) == model.L * 5

        # for all classes
        for progress, dp in enumerate(data_points):

            # progress display
            vs = "Processing datapoint " + str(progress + 1) + \
                " / " + str(len(data_points)) + "\n"
            sys.stdout.write(vs)
            sys.stdout.flush()

            x, y = dp

            # for all classes except the true class
            for c in range(model.L):

                if c != y.item():

                    y_target = torch.tensor([c])

                    # keep adv_x as close to x as possible while making the network think it
                    # belongs to the class y_target with high probability
                    # this classification is based on norm; we restrict the norm to 1
                    advex = model.get_adversary(x = x,
                                                y = y,
                                                adv_x = x,
                                                y_target = y_target,
                                                #converge_fn = lambda target_p, diff_norm: diff_norm > 1.0 or target_p > 95.0,
                                                converge_fn = lambda target_p, diff_norm: target_p > 50.0,
                                                debug = True)
                    adv_examples.append(advex)

        # store adversarial examples in a folder
        if report_folder != "":
            assert folder_exists(report_folder)

            # create a folder to store adv examples
            advex_store = os.path.join(report_folder, model.desc + "_advex")
            create_folder(advex_store)
            assert folder_exists(advex_store)

            # pickle advexamples to folder
            advex_store_file = os.path.join(advex_store, "advex.pkl")
            pickle_f = open(advex_store_file, "w+")
            pickle.dump(adv_examples, pickle_f)
            pickle_f.close()

################################################################################

####

# train_mnist()

model_paths_mnist_img = ["./checkpoints/mnist_image_models/MNIST_img_Logistic_1L_2018-11-21_19-06-20/generic_Logistic_1L_Tr_acc_0.9249_Ts_acc_0.9245/model.pkl",
            "./checkpoints/mnist_image_models/MNIST_img_Logistic_2L_2018-11-21_17-49-25/generic_Logistic_2L_Tr_acc_0.92145_Ts_acc_0.919/model.pkl",
            "./checkpoints/mnist_image_models/MNIST_img_Logistic_LRL_2018-11-21_17-41-15/generic_Logistic_LRL_Tr_acc_0.940083333333_Ts_acc_0.94/model.pkl"]

model_paths_fmnist_img = ["./checkpoints/fmnist_image_models/FMNIST_img_Logistic_1L_2018-11-23_15-17-46/generic_Logistic_1L_Tr_acc_0.860516666667_Ts_acc_0.8401/model.pkl",
            "./checkpoints/fmnist_image_models/FMNIST_img_Logistic_2L_2018-11-23_15-08-36/generic_Logistic_2L_Tr_acc_0.860766666667_Ts_acc_0.8419/model.pkl",
            "./checkpoints/fmnist_image_models/FMNIST_img_Logistic_LRL_2018-11-23_14-59-20/generic_Logistic_LRL_Tr_acc_0.86835_Ts_acc_0.8472/model.pkl"]

model_paths_mnist_ae = ["./checkpoints/AE_MNIST_MODELS/Logistic_1L_2018-11-18_09-59-24/MNIST train dataset_Logistic_1L_Tr_acc_0.892416666667_Ts_acc_0.8984/model.pkl",
        "./checkpoints/AE_MNIST_MODELS/Logistic_2L_2018-11-18_14-36-17/MNIST train dataset_Logistic_2L_Tr_acc_0.891316666667_Ts_acc_0.8945/model.pkl",
        "./checkpoints/AE_MNIST_MODELS/Logistic_LRL_2018-11-18_05-22-27/MNIST train dataset_Logistic_LRL_Tr_acc_0.96355_Ts_acc_0.9579/model.pkl"]

model_paths_mnist_vae = ["./checkpoints/VAE_MNIST_MODELS/Logistic_1L_2018-11-19_17-40-41/VAE MNIST train dataset_Logistic_1L_Tr_acc_0.851433333333_Ts_acc_0.8573/model.pkl",
              "./checkpoints/VAE_MNIST_MODELS/Logistic_2L_2018-11-19_19-24-35/VAE MNIST train dataset_Logistic_2L_Tr_acc_0.847316666667_Ts_acc_0.8548/model.pkl",
              "./checkpoints/VAE_MNIST_MODELS/Logistic_LRL_2018-11-19_14-50-32/VAE MNIST train dataset_Logistic_LRL_Tr_acc_0.9279_Ts_acc_0.9292/model.pkl"]

model_paths_fmnist_ae = ["./checkpoints/AE_FMNIST_MODELS/Logistic_1L_2018-11-19_04-00-11/MNIST train dataset_Logistic_1L_Tr_acc_0.813466666667_Ts_acc_0.7968/model.pkl",
               "./checkpoints/AE_FMNIST_MODELS/Logistic_2L_2018-11-19_08-32-20/MNIST train dataset_Logistic_2L_Tr_acc_0.81425_Ts_acc_0.8005/model.pkl",
               "./checkpoints/AE_FMNIST_MODELS/Logistic_LRL_2018-11-18_21-56-57/MNIST train dataset_Logistic_LRL_Tr_acc_0.85545_Ts_acc_0.8375/model.pkl"]

model_paths_fmnist_vae = ["./checkpoints/VAE_FMNIST_MODELS/Logistic_1L_2018-11-20_02-33-03/VAE FMNIST train dataset_Logistic_1L_Tr_acc_0.7444_Ts_acc_0.7371/model.pkl",
        "./checkpoints/VAE_FMNIST_MODELS/Logistic_2L_2018-11-20_04-17-20/VAE FMNIST train dataset_Logistic_2L_Tr_acc_0.7575_Ts_acc_0.7463/model.pkl",
        "./checkpoints/VAE_FMNIST_MODELS/Logistic_LRL_2018-11-20_00-40-34/VAE FMNIST train dataset_Logistic_LRL_Tr_acc_0.804583333333_Ts_acc_0.7966/model.pkl"]

## generate advex

generate_advex(model_paths_mnist_img, "./mnist_img_advex")

generate_advex(model_paths_fmnist_img, "./fmnist_img_advex")

generate_advex(model_paths_mnist_ae, "./mnist_ae_advex")

generate_advex(model_paths_mnist_vae, "./mnist_vae_advex")

generate_advex(model_paths_fmnist_ae, "./fmnist_ae_advex")

generate_advex(model_paths_fmnist_vae, "./fmnist_vae_advex")
