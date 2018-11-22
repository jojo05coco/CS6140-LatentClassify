
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

def generate_advex(report_folder = ""):

    if report_folder != "":
        delete_folder(report_folder)
        create_folder(report_folder)
        assert folder_exists(report_folder)


    model_paths = ["./mnist_image_models/MNIST_img_Logistic_1L_2018-11-21_19-06-20/generic_Logistic_1L_Tr_acc_0.9249_Ts_acc_0.9245/model.pkl",
            "./mnist_image_models/MNIST_img_Logistic_2L_2018-11-21_17-49-25/generic_Logistic_2L_Tr_acc_0.92145_Ts_acc_0.919/model.pkl",
            "./mnist_image_models/MNIST_img_Logistic_LRL_2018-11-21_17-41-15/generic_Logistic_LRL_Tr_acc_0.940083333333_Ts_acc_0.94/model.pkl"]

    for model_path in model_paths:

        adv_examples = []

        # model
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
                                                converge_fn = lambda target_p, diff_norm: diff_norm > 1.0,
                                                debug = False)
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

generate_advex("mnist_advex")

# model_1l path
#model_1l_path = "./mnist_image_models/MNIST_img_Logistic_1L_2018-11-21_19-06-20/generic_Logistic_1L_Tr_acc_0.9249_Ts_acc_0.9245/model.pkl"
#
## load mnist image model; load the 1 layer model first
#mf = open(model_1l_path, "rb")
#model_1l = pickle.load(mf)
#mf.close()
#
## loaded model
#print "Model description ", model_1l.desc
#
## get the first element and target from the test set
#
## Get the first test sample from the model
#x_target, y = model_1l.test_ds[1]
#x = x_target
#y_target = torch.tensor([3])
#
## keep adv_x as close to x as possible while making the network think it
## belongs to the class y_target with high probability
#advex = model_1l.get_adversary(x = x,
#                               y = y,
#                               adv_x = x,
#                               y_target = y_target,
#                               converge_fn = lambda target_p, diff_norm: diff_norm > 1.0,
#                               debug = False)
#
#print "\n\n Adversarial Example : ", advex
#
#advex.show()
#
##clf.train()
