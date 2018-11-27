
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
from svm_classifier import SVM_Classifier
from dataset import CLF_DS

from file_utils import create_folder, delete_folder, folder_exists
from utils import random_data_points

# folder containing/to-contain the dataset
root = "./data"

## Utils #######################################################################

def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def train_mnist():

    #trans = transforms.Compose([transforms.ToTensor()])
    #train_set = dset.MNIST(root = root, train = True, transform = trans, download = True)
    #test_set = dset.MNIST(root = root, train = False, transform = trans, download = True)

    #print "Train feature set shape ", train_set.train_data.shape
    #print "Train labels set shape ", train_set.train_labels.shape

    ## save the data
    ## train features
    #train_features = train_set.train_data.numpy()
    #train_labels   = train_set.train_labels.numpy()
    #test_features  = test_set.test_data.numpy()
    #test_labels    = test_set.test_labels.numpy()

    #train_features = np.array(map(lambda img: img.flatten(), train_features))
    #test_features = np.array(map(lambda img: img.flatten(), test_features))

    ## shapes
    #print "Train features shape ", train_features.shape
    #print "Train labels shape ", train_labels.shape
    #print "Test features shape ", test_features.shape
    #print "Test labels shape ", test_labels.shape

    #f = open("./test/IMG_MNIST_image_tr.npy", "wb+")
    #np.save(f, train_features)
    #f.close()

    #f = open("./test/IMG_MNIST_targets_tr.npy", "wb+")
    #np.save(f, train_labels)
    #f.close()

    #f = open("./test/IMG_MNIST_image_ts.npy", "wb+")
    #np.save(f, test_features)
    #f.close()

    #f = open("./test/IMG_MNIST_targets_ts.npy", "wb+")
    #np.save(f, test_labels)
    #f.close()

    train_x = "./test/IMG_MNIST_image_tr.npy"
    train_y = "./test/IMG_MNIST_targets_tr.npy"
    test_x = "./test/IMG_MNIST_image_ts.npy"
    test_y = "./test/IMG_MNIST_targets_ts.npy"

    Cs = [1, 10, 20, 50]

    for c in Cs:
        report_folder = "MNIST_IMG_SVM_" + str(c)
        svm_clf = SVM_Classifier(train_x,
                                 train_y,
                                 test_x,
                                 test_y,
                                 C = c,
                                 report_folder = report_folder)

        svm_clf.train()
        svm_clf.report()

    #train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = 1, shuffle = True)

    #clf = Logistic_LRL(train_set, test_set, batch_size = 1000, epochs = 150, n_h1 = 30,
    #                   report_folder = "Logistic_LRL_" + timestamp(), is_CLF_DS = False)
    #clf.train()

    #clf = Logistic_2L(train_set, test_set, batch_size = 1000, epochs = 50, n_h1 = 30,
    #                   report_folder = "MNIST_img_Logistic_2L_" + timestamp(), is_CLF_DS = False)

    #clf.train()

    #clf = Logistic_1L(train_set, test_set, batch_size = 1000, epochs = 50,
    #                   report_folder = "MNIST_img_Logistic_LRL_" + timestamp(), is_CLF_DS = False)
    #clf.train()

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
        data_points = random_data_points(model, model.test_ds, model.L, 2)
        print "Model L ", model.L, " | ", len(data_points)
        assert len(data_points) == model.L * 2

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


        # how many non None examples did we get ?
        n_none = len(filter(lambda ex: ex is None, adv_examples))
        print "\n", len(adv_examples) - n_none, " / ", len(adv_examples), " Valid "
        # How many samples under tolerence
        n_tol = len(filter(lambda ex: ex is not None and ex.adv_diff_norm < 51.0, adv_examples))
        print n_tol, " / ", len(adv_examples), " tolerant samples"

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

#train_mnist()

#exit(-1)

# train_mnist()

model_paths_mnist_HD = ["./MNIST_HD_img_Logistic_2L_2018-11-26_17-15-34/generic_Logistic_2L_Tr_acc_0.927566666667_Ts_acc_0.9258/model.pkl"]

model_paths_mnist_img = ["./checkpoints/mnist_image_models/MNIST_img_Logistic_1L_2018-11-21_19-06-20/generic_Logistic_1L_Tr_acc_0.9249_Ts_acc_0.9245/model.pkl",
                        "./checkpoints/mnist_image_models/MNIST_img_Logistic_2L_2018-11-21_17-49-25/generic_Logistic_2L_Tr_acc_0.92145_Ts_acc_0.919/model.pkl",
                        "./checkpoints/mnist_image_models/MNIST_img_Logistic_LRL_2018-11-21_17-41-15/generic_Logistic_LRL_Tr_acc_0.940083333333_Ts_acc_0.94/model.pkl"]

model_paths_fmnist_img = ["./checkpoints/fmnist_image_models/FMNIST_img_Logistic_1L_2018-11-23_15-17-46/generic_Logistic_1L_Tr_acc_0.860516666667_Ts_acc_0.8401/model.pkl",
                          "./checkpoints/fmnist_image_models/FMNIST_img_Logistic_2L_2018-11-23_15-08-36/generic_Logistic_2L_Tr_acc_0.860766666667_Ts_acc_0.8419/model.pkl",
                          "./checkpoints/fmnist_image_models/FMNIST_img_Logistic_LRL_2018-11-23_14-59-20/generic_Logistic_LRL_Tr_acc_0.86835_Ts_acc_0.8472/model.pkl"]

model_paths_mnist_ae = ["./checkpoints/MNIST_AE_MODELS/MNIST_AELogistic_1L_2018-11-25_21-44-35/MNIST_AE_Logistic_1L_Tr_acc_0.893133333333_Ts_acc_0.8985/model.pkl",
                        "./checkpoints/MNIST_AE_MODELS/MNIST_AELogistic_2L_2018-11-25_21-47-51/MNIST_AE_Logistic_2L_Tr_acc_0.8915_Ts_acc_0.8982/model.pkl",
                        "./checkpoints/MNIST_AE_MODELS/MNIST_AE_Logistic_LRL_2018-11-25_21-40-55/MNIST_AE_Logistic_LRL_Tr_acc_0.964183333333_Ts_acc_0.9584/model.pkl"]

model_paths_mnist_vae = ["./checkpoints/MNIST_VAE_MODELS/MNIST_VAELogistic_1L_2018-11-25_21-54-44/MNIST_VAE_Logistic_1L_Tr_acc_0.847383333333_Ts_acc_0.8537/model.pkl",
                         "./checkpoints/MNIST_VAE_MODELS/MNIST_VAELogistic_2L_2018-11-25_21-57-54/MNIST_VAE_Logistic_2L_Tr_acc_0.844766666667_Ts_acc_0.8482/model.pkl",
                         "./checkpoints/MNIST_VAE_MODELS/MNIST_VAE_Logistic_LRL_2018-11-25_21-51-16/MNIST_VAE_Logistic_LRL_Tr_acc_0.9332_Ts_acc_0.9287/model.pkl"]

model_paths_mnist_ae_l2 = ["./checkpoints/MNIST_AE_L2_MODELS/MNIST_AE_L2Logistic_1L_2018-11-25_22-05-01/MNIST_AE_L2_Logistic_1L_Tr_acc_0.845166666667_Ts_acc_0.8514/model.pkl",
                          "./checkpoints/MNIST_AE_L2_MODELS/MNIST_AE_L2Logistic_2L_2018-11-25_22-08-19/MNIST_AE_L2_Logistic_2L_Tr_acc_0.840616666667_Ts_acc_0.8454/model.pkl",
                          "./checkpoints/MNIST_AE_L2_MODELS/MNIST_AE_L2_Logistic_LRL_2018-11-25_22-01-26/MNIST_AE_L2_Logistic_LRL_Tr_acc_0.94495_Ts_acc_0.9374/model.pkl"]

model_paths_mnist_vae_l2 = ["./checkpoints/MNIST_VAE_L2_MODELS/MNIST_VAE_L2Logistic_1L_2018-11-25_22-15-30/MNIST_VAE_L2_Logistic_1L_Tr_acc_0.81295_Ts_acc_0.8197/model.pkl",
                            "./checkpoints/MNIST_VAE_L2_MODELS/MNIST_VAE_L2Logistic_2L_2018-11-25_22-18-50/MNIST_VAE_L2_Logistic_2L_Tr_acc_0.807983333333_Ts_acc_0.8155/model.pkl",
                            "./checkpoints/MNIST_VAE_L2_MODELS/MNIST_VAE_L2_Logistic_LRL_2018-11-25_22-11-51/MNIST_VAE_L2_Logistic_LRL_Tr_acc_0.91205_Ts_acc_0.9092/model.pkl"]

model_paths_fmnist_ae = ["./checkpoints/FMNIST_AE_MODELS/FMNIST_AELogistic_1L_2018-11-25_22-26-19/FMNIST_AE_Logistic_1L_Tr_acc_0.816666666667_Ts_acc_0.8068/model.pkl",
                         "./checkpoints/FMNIST_AE_MODELS/FMNIST_AELogistic_2L_2018-11-25_22-30-03/FMNIST_AE_Logistic_2L_Tr_acc_0.82645_Ts_acc_0.812/model.pkl",
                         "./checkpoints/FMNIST_AE_MODELS/FMNIST_AE_Logistic_LRL_2018-11-25_22-22-21/FMNIST_AE_Logistic_LRL_Tr_acc_0.857916666667_Ts_acc_0.8382/model.pkl"]

model_paths_fmnist_vae = ["./checkpoints/FMNIST_VAE_MODELS/FMNIST_VAELogistic_1L_2018-11-25_22-37-32/FMNIST_VAE_Logistic_1L_Tr_acc_0.758166666667_Ts_acc_0.7472/model.pkl",
                          "./checkpoints/FMNIST_VAE_MODELS/FMNIST_VAELogistic_2L_2018-11-25_22-40-56/FMNIST_VAE_Logistic_2L_Tr_acc_0.760983333333_Ts_acc_0.7516/model.pkl",
                          "./checkpoints/FMNIST_VAE_MODELS/FMNIST_VAE_Logistic_LRL_2018-11-25_22-33-58/FMNIST_VAE_Logistic_LRL_Tr_acc_0.812016666667_Ts_acc_0.794/model.pkl"]

model_paths_fmnist_ae_l2 = ["./checkpoints/FMNIST_AE_L2_MODELS/FMNIST_AE_L2Logistic_1L_2018-11-25_22-48-06/FMNIST_AE_L2_Logistic_1L_Tr_acc_0.762583333333_Ts_acc_0.751/model.pkl",
                            "./checkpoints/FMNIST_AE_L2_MODELS/FMNIST_AE_L2Logistic_2L_2018-11-25_22-51-18/FMNIST_AE_L2_Logistic_2L_Tr_acc_0.759566666667_Ts_acc_0.7484/model.pkl",
                            "./checkpoints/FMNIST_AE_L2_MODELS/FMNIST_AE_L2_Logistic_LRL_2018-11-25_22-44-33/FMNIST_AE_L2_Logistic_LRL_Tr_acc_0.8097_Ts_acc_0.7931/model.pkl"]

model_paths_fmnist_vae_l2 = ["./checkpoints/FMNIST_VAE_L2_MODELS/FMNIST_VAE_L2Logistic_1L_2018-11-25_22-58-16/FMNIST_VAE_L2_Logistic_1L_Tr_acc_0.686916666667_Ts_acc_0.6768/model.pkl",
                             "./checkpoints/FMNIST_VAE_L2_MODELS/FMNIST_VAE_L2Logistic_2L_2018-11-25_23-01-43/FMNIST_VAE_L2_Logistic_2L_Tr_acc_0.680983333333_Ts_acc_0.6684/model.pkl",
                             "./checkpoints/FMNIST_VAE_L2_MODELS/FMNIST_VAE_L2_Logistic_LRL_2018-11-25_22-54-42/FMNIST_VAE_L2_Logistic_LRL_Tr_acc_0.766366666667_Ts_acc_0.7502/model.pkl"]

## generate advex

#generate_advex(model_paths_mnist_HD, "./MNIST_HD_advex")

#generate_advex(model_paths_mnist_img,     "./adversary_dataset/CONFIDENCE_75/mnist_img_75conf_advex")

#generate_advex(model_paths_fmnist_img,    "./adversary_dataset/CONFIDENCE_75/fmnist_img_75conf_advex")

#generate_advex(model_paths_mnist_vae,     "./adversary_dataset/CONFIDENCE_50/mnist_vae_50conf_advex")

#generate_advex(model_paths_fmnist_vae,    "./adversary_dataset/CONFIDENCE_50/fmnist_vae_50conf_advex")

#generate_advex(model_paths_mnist_ae,      "./adversary_dataset/CONFIDENCE_95/mnist_ae_95conf_advex")
#
#generate_advex(model_paths_mnist_vae,     "./adversary_dataset/CONFIDENCE_95/mnist_vae_95conf_advex")
#
#generate_advex(model_paths_mnist_ae_l2,   "./adversary_dataset/CONFIDENCE_95/mnist_ae_l2_95conf_advex")
#
#generate_advex(model_paths_mnist_vae_l2,  "./adversary_dataset/CONFIDENCE_95/mnist_vae_l2_95conf_advex")
#
#generate_advex(model_paths_fmnist_ae,     "./adversary_dataset/CONFIDENCE_95/fmnist_ae_95conf_advex")
#
#generate_advex(model_paths_fmnist_vae,    "./adversary_dataset/CONFIDENCE_95/fmnist_vae_95conf_advex")
#
#generate_advex(model_paths_fmnist_ae_l2,  "./adversary_dataset/CONFIDENCE_95/fmnist_ae_l2_95conf_advex")
#
#generate_advex(model_paths_fmnist_vae_l2, "./adversary_dataset/CONFIDENCE_95/fmnist_vae_l2_95conf_advex")

exit(-1)
