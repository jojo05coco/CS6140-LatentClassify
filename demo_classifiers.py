## Demo program on how to use the classifiers

import os
import datetime

from dataset import CLF_DS

from classifiers import Logistic_1L, Logistic_2L, Logistic_LRL, Logistic_LRLRL

from svm_classifier import SVM_Classifier

## Utils #######################################################################

def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# latent codes train path
#latent_codes_train = "./test/AE_MNIST-latent_tr.pth"
#targets_train      = "./test/AE_MNIST-targets_tr.pth"
#latent_codes_test  = "./test/AE_MNIST-latent_ts.pth"
#targets_test       = "./test/AE_MNIST-targets_ts.pth"

#latent_codes_train = "./test/AE_FMNIST_latent_tr.pth"
#targets_train      = "./test/AE_FMNIST_targets_tr.pth"
#latent_codes_test  = "./test/AE_FMNIST_latent_ts.pth"
#targets_test       = "./test/AE_FMNIST_targets_ts.pth"

#latent_codes_train = "./test/VAE_MNIST-latent_tr.pth"
#targets_train      = "./test/VAE_MNIST-targets_tr.pth"
#latent_codes_test  = "./test/VAE_MNIST-latent_ts.pth"
#targets_test       = "./test/VAE_MNIST-targets_ts.pth"

#latent_codes_train = "./test/VAE_FMNIST-latent_tr.pth"
#targets_train      = "./test/VAE_FMNIST-targets_tr.pth"
#latent_codes_test  = "./test/VAE_FMNIST-latent_ts.pth"
#targets_test       = "./test/VAE_FMNIST-targets_ts.pth"

#latent_codes_train = "./test/AE_MNIST_L2latent_tr.npy"
#targets_train      = "./test/AE_MNIST-targets_tr.pth"
#latent_codes_test  = "./test/AE_MNIST_L2latent_ts.npy"
#targets_test       = "./test/AE_MNIST-targets_ts.pth"

#latent_codes_train = "./test/AE_MNIST_L2latent_tr.npy"
#targets_train      = "./test/AE_MNIST-targets_tr.pth"
#latent_codes_test  = "./test/AE_MNIST_L2latent_ts.npy"
#targets_test       = "./test/AE_MNIST-targets_ts.pth"

latent_codes_train = "./test/AE_FMNIST_L2latent_tr.npy"
targets_train      = "./test/AE_FMNIST_targets_tr.pth"
latent_codes_test  = "./test/AE_FMNIST_L2latent_ts.npy"
targets_test       = "./test/AE_FMNIST_targets_ts.pth"

## Neural networks #############################################################

train_ds = CLF_DS(latent_codes_train, targets_train, "Projection AE MNIST train dataset")
test_ds  = CLF_DS(latent_codes_test, targets_test, "Projection AE MNIST test dataset")

clf = Logistic_LRL(train_ds, test_ds, batch_size = 1000, epochs = 150,
                   report_folder = "Logistic_LRL_" + timestamp())
clf.train()

clf = Logistic_1L(train_ds, test_ds, batch_size = 1000, epochs = 150,
                  report_folder = "Logistic_1L_" + timestamp())
clf.train()

clf = Logistic_2L(train_ds, test_ds, batch_size = 1000, epochs = 150,
                  report_folder = "Logistic_2L_" + timestamp())
clf.train()

## SVM classifier ##############################################################

Cs = [1, 10, 20, 50]

for c in Cs:
    report_folder = "SVM_" + str(c)
    svm_clf = SVM_Classifier(latent_codes_train,
                             targets_train,
                             latent_codes_test,
                             targets_test,
                             C = c,
                             report_folder = report_folder)

    svm_clf.train()
    svm_clf.report()
