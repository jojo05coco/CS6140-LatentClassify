## Use svm to classify data; svm from sklearn is used

import os

import numpy as np
import torch # for loading train and test data
import pickle

from sklearn import svm

from file_utils import create_folder, delete_folder, folder_exists

class SVM_Classifier:

    ## Constructor
    # train_x_pth : Path to the training latent codes .pth/.npy file
    # train_y_pth : Path to the training targets .pth/.npy file
    # test_x_pth  : Path to the test latent codes .pth/.npy file
    # test_y_pth  : Path to the test targets .pth/.npy file
    def __init__(self, train_x_pth, train_y_pth, test_x_pth, test_y_pth, kernel = 'rbf', C = 1, report_folder = ""):

        # record report folder
        self.report_folder = report_folder
        # sanitize
        if self.report_folder != "":
            delete_folder(self.report_folder)
            create_folder(self.report_folder)
            assert folder_exists(self.report_folder)

        if train_x_pth.endswith("npy"):
            self.train_x = np.load(train_x_pth)
        else:
            self.train_x = torch.load(train_x_pth).detach().numpy()

        if train_y_pth.endswith("npy"):
            self.train_y = np.load(train_y_pth)
        else:
            self.train_y = torch.load(train_y_pth).detach().numpy()

        if test_x_pth.endswith("npy"):
            self.test_x  = np.load(test_x_pth)
        else:
            self.test_x  = torch.load(test_x_pth).detach().numpy()

        if test_y_pth.endswith("npy"):
            self.test_y  = np.load(test_y_pth)
        else:
            self.test_y  = torch.load(test_y_pth).detach().numpy()

        #self.train_x = self.train_x[0:2000]
        #self.train_y = self.train_y[0:2000]

        # dataset information
        self.N = self.train_x.shape[0]
        self.D = self.train_x.shape[1]
        self.NTest = self.test_x.shape[0]

        # sanity check
        assert self.train_y.shape[0] == self.N
        assert self.test_y.shape[0]  == self.NTest
        assert self.test_x.shape[1]  == self.D

        # kernel to use
        self.kernel = kernel
        # regularization parameter
        self.C      = C

        self.model = svm.SVC(C = self.C,
                             kernel = self.kernel,
                             decision_function_shape = 'ovo')

    # train
    def train(self):

        print "Training ..."

        self.model.fit(self.train_x, self.train_y)

        # print train and test loss
        print "SVM : C = ", str(self.C), ", kernel = ", self.kernel, \
              " | Train acc. : ", self.train_accuracy(),             \
              " | Test acc. : ", self.test_accuracy()

    # train accuracy
    def train_accuracy(self):

        pred_ys = np.array(map(lambda x: self.model.predict([x]).item(), self.train_x))

        fail_arr = pred_ys - self.train_y

        n_fails = np.count_nonzero(fail_arr)

        n_correct = self.N - n_fails

        return float(n_correct) / float(self.N)

    # test accuracy
    def test_accuracy(self):

        pred_ys = np.array(map(lambda x: self.model.predict([x]).item(), self.test_x))

        fail_arr = pred_ys - self.test_y

        n_fails = np.count_nonzero(fail_arr)

        n_correct = self.NTest - n_fails

        return float(n_correct) / float(self.NTest)

    def test(self, x):

        x = x.detach().numpy()

        pred = self.model.predict([x])

        return pred

    # report
    def report(self):

        assert self.report_folder != ""
        assert folder_exists(self.report_folder)

        info_s = "SVM : C = " + str(self.C) + ", kernel = " + self.kernel + \
              " | Train acc. : " + str(self.train_accuracy()) +             \
              " | Test acc. : " + str(self.test_accuracy())

        # Info file name
        info_file_name = os.path.join(self.report_folder, "info.txt")
        # write info to file
        info_f = open(info_file_name, "w+")
        info_f.write(info_s)
        info_f.close()

        # pickle self
        pckle_file_name = os.path.join(self.report_folder, "model.pkl")
        # write self to file
        pckle_f = open(pckle_file_name, "w+")
        pickle.dump(self, pckle_f)
        pckle_f.close()

################################################################################
