## All classifier classes

import os
import sys

import numpy as np
import torch
import pickle

from file_utils import delete_folder, create_folder, folder_exists

VERBOSE = 1

class Classifier:

    ## Constructor
    # train_ds : train dataset
    # test_ds  : test dataset
    # batch size : train batch size
    def __init__(self,
                 train_ds,
                 test_ds,
                 batch_size = 100,
                 nepochs = 1000,
                 report_folder = "",
                 desc = ""):

        # record inputs in self
        self.train_ds      = train_ds
        self.test_ds       = test_ds
        self.batch_size    = batch_size
        self.nepochs       = nepochs
        self.report_folder = report_folder
        self.desc          = desc

        # initialize epoch
        self.epoch = 0

        # update intervals; in units for epoch
        self.update_interval = 1
        # report gen interval; report every 10 epochs
        self.report_interval = 10

        # print information
        print self.train_ds
        print self.test_ds

        # Sanity check
        assert self.test_ds.D == self.train_ds.D
        assert self.test_ds.L == self.train_ds.L

        # features
        self.D = train_ds.D
        self.L = self.test_ds.L

        # Lets create data loaders
        self.train_loader = torch.utils.data.DataLoader(self.train_ds,
                                                        batch_size = self.batch_size,
                                                        shuffle = True)

        # report folder setup
        if self.report_folder != "":
            delete_folder(self.report_folder)
            create_folder(self.report_folder)

        # All classes that inherit must have a model variable that is a torch module
        #self.model
        #self.optimizer

    def train(self):

        # for nepochs
        while self.epoch < self.nepochs:

            # inc. self.epoch
            self.epoch = self.epoch + 1

            self.model.train()

            # through all your datapoints
            for bidx, (data, targets) in enumerate(self.train_loader):

                # make 2D target tensor; 1D
                targets = torch.tensor(targets.numpy().flatten())

                # setup
                self.optimizer.zero_grad()

                # forward
                log_softmax = self.model(data)

                # prediction
                loss = torch.nn.functional.nll_loss(log_softmax, targets)

                #pred_targets = log_softmax[0].max(0)[1]
                #print "Log softmax ", log_softmax
                #print "Targets ", targets
                #print "Pred Targets ", pred_targets
                #nfail = np.count_nonzero((targets - pred_targets).numpy())

                # back prop
                loss.backward()

                #for param in model.parameters():
                #    print param.grad

                # update weights
                self.optimizer.step()

                if VERBOSE:
                    verbose_s = "Epoch " + str(self.epoch) + ", Batch " + str(bidx) + " done. \r"
                    sys.stdout.write(verbose_s)
                    sys.stdout.flush()

            if self.epoch % self.update_interval == 0:

                train_loss, train_acc = self.train_loss()
                test_loss, test_acc = self.test_loss()

                print " Train Epoch : ", self.epoch,  \
                      " | Train loss : ", train_loss, \
                      " | Test loss : ", test_loss,   \
                      " | Train acc. : ", train_acc,  \
                      " | Test acc. : ", test_acc

            if self.epoch % self.report_interval == 0:
                self.report()

    def test(self, x):

        assert len(x) == self.train_ds.D

        model.eval()

        log_softmax = model(x)

        prediction = log_softmax[0].max(0)[1]

        return prediction

    def train_loss(self):
        return self.loss(self.train_ds)

    def test_loss(self):
        return self.loss(self.test_ds)

    # evaluate loss on the given dataset
    def loss(self, ds):

        dloader = torch.utils.data.DataLoader(ds,
                                              batch_size = len(ds),
                                              shuffle = False)

        acc = 0.0
        l   = 0.0

        ncorrect = 0

        self.model.eval

        with torch.no_grad():

            for data, targets in dloader:

                # make 2D target tensor; 1D
                targets = torch.tensor(targets.numpy().flatten())

                log_softmax = self.model(data)

                loss = torch.nn.functional.nll_loss(log_softmax, targets)

                pred_targets = torch.tensor(np.array(map(lambda r: r.max(0)[1], log_softmax)))

                target_diff = targets - pred_targets

                nfail = np.count_nonzero((target_diff).numpy())

                ncorrect = ncorrect + (len(targets) - nfail)

                l = l + loss.item()

        acc = float(ncorrect) / float(len(ds))

        return l, acc

    def report(self):

        assert folder_exists(self.report_folder)

        # get test and train accuracy
        train_acc = "Tr_acc_" + str(self.train_loss()[1])
        test_acc  = "Ts_acc_" + str(self.test_loss()[1])

        # create a folder to store current state
        report_folder_now = os.path.join(self.report_folder,
                                         self.train_ds.desc + "_" + \
                                         self.desc + "_" + \
                                         train_acc + "_" + \
                                         test_acc)
        delete_folder(report_folder_now)
        create_folder(report_folder_now)
        assert folder_exists(report_folder_now)

        # Info file name
        info_file_name = os.path.join(report_folder_now, "info.txt")
        info_str = self.__str__()
        # write info to file
        info_f = open(info_file_name, "w+")
        info_f.write(info_str)
        info_f.close()

        # pickle self
        pckle_file_name = os.path.join(report_folder_now, "model.pkl")
        # write self to file
        pckle_f = open(pckle_file_name, "w+")
        pickle.dump(self, pckle_f)
        pckle_f.close()


    def __str__(self):

        # represent self in a string
        s = "========================================================\n"
        s = s + " Model : " + self.desc + "\n"
        s = s + " Dataset Train - " + self.train_ds.desc   + "\n"
        s = s + " Dataset Test  - " + self.test_ds.desc    + "\n"
        s = s + " Batch size    - " + str(self.batch_size) + "\n"
        s = s + " nepochs       - " + str(self.nepochs)    + "\n"
        s = s + " epoch         - " + str(self.epoch)      + "\n"
        s = s + " Train acc.    - " + str(self.train_loss()) + "\n"
        s = s + " Test acc.     - " + str(self.test_loss()) + "\n"
        s = s + "========================================================\n"

        return s

    def __repr__(self):
        return self.__str__()

################################################################################

# Model : linear-layer -> softmax
class Logistic_1L(Classifier):

    ## Constructor
    def __init__(self,
                 train_ds,
                 test_ds,
                 learning_rate = 0.0005,
                 batch_size = 100,
                 epochs = 1000,
                 report_folder = "",
                 desc = "Logistic_1L"):

        # let parent do all the work
        Classifier.__init__(self,
                            train_ds,
                            test_ds,
                            batch_size = batch_size,
                            nepochs = epochs,
                            report_folder = report_folder,
                            desc = desc)

        self.model     = torch.nn.Sequential(torch.nn.Linear(self.D, self.L),
                                             torch.nn.LogSoftmax(1))
        # optimizer
        #self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                 lr = learning_rate,
        #                                 momentum = 0.9,
        #                                 weight_decay = 1.0)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)


################################################################################

# Model linear-layer -> linear-layer -> softmax
class Logistic_2L(Classifier):

    ## Constructor
    def __init__(self,
                 train_ds,
                 test_ds,
                 learning_rate = 0.0005,
                 batch_size = 100,
                 epochs = 1000,
                 report_folder = "",
                 desc = "Logistic_2L"):

        # let parent do all the work
        Classifier.__init__(self,
                            train_ds,
                            test_ds,
                            batch_size = batch_size,
                            nepochs = epochs,
                            report_folder = report_folder,
                            desc = desc)

        # hidden layer 1 dimensions;
        self.n_h1 = self.D + 50 # Arbitrary

        self.model     = torch.nn.Sequential(torch.nn.Linear(self.D, self.n_h1),
                                             torch.nn.Dropout(0.2),
                                             torch.nn.Linear(self.n_h1, self.L),
                                             torch.nn.LogSoftmax(1))
        # optimizer
        #self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                 lr = learning_rate,
        #                                 momentum = 0.9,
        #                                 weight_decay = 1.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

##################################################################################

# Model linear-layer -> LeakyRelu -> linear-layer -> softmax
class Logistic_LRL(Classifier):

    ## Constructor
    def __init__(self,
                 train_ds,
                 test_ds,
                 learning_rate = 0.0005,
                 batch_size = 100,
                 epochs = 1000,
                 report_folder = "",
                 desc = "Logistic_LRL"):

        # let parent do all the work
        Classifier.__init__(self,
                            train_ds,
                            test_ds,
                            batch_size = batch_size,
                            nepochs = epochs,
                            report_folder = report_folder,
                            desc = desc)

        # hidden layer 1 dimensions;
        self.n_h1 = self.D + 50 # Arbitrary

        self.model     = torch.nn.Sequential(torch.nn.Linear(self.D, self.n_h1),
                                             torch.nn.LeakyReLU(),
                                             torch.nn.Dropout(0.2),
                                             torch.nn.Linear(self.n_h1, self.L),
                                             torch.nn.LogSoftmax(1))
        # optimizer
        #self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                 lr = learning_rate,
        #                                 momentum = 0.9,
        #                                 weight_decay = 1.0)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
