## Use svm to classify data; svm from sklearn is used

import numpy as np
import torch # for loading train and test data

from sklearn import svm

class SVM_Classifier:

    ## Constructor
    # train_x_pth : Path to the training latent codes .pth file
    # train_y_pth : Path to the training targets .pth file
    # test_x_pth  : Path to the test latent codes .pth file
    # test_y_pth  : Path to the test targets .pth file
    def __init__(self, train_x_pth, train_y_pth, test_x_pth, test_y_pth, kernel = 'rbf', C = 1):

        self.train_x = torch.load(train_x_pth).detach().numpy()
        self.train_y = torch.load(train_y_pth).detach().numpy()
        self.test_x  = torch.load(test_x_pth).detach().numpy()
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
