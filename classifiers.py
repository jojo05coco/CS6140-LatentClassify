## All classifier classes

import os
import sys
import math

import matplotlib.pyplot as plt

import numpy as np
import torch
import pickle

from file_utils import delete_folder, create_folder, folder_exists
from utils import l2

VERBOSE = 1

## Adversary datastructure #
class AdversarialExample:

    ## Constructor
    #
    # x           : original image / latent code
    # y           : true label
    # y_target    : we want model to think x belongs to y_target with small pertubation.
    #               i.e. the adversarial example must be classified to be in class y_target
    # adv_x       : x's adversarial example
    # x_probs     : model probabilities with x as input
    # adv_x_probs : model probabilities with adv_x as input
    # model_desc  : description of model used
    def __init__(self, x, y, adv_x, y_target, x_probs, adv_x_probs, adv_diff_norm, model_desc):

        self.x             = x
        self.y             = y
        self.adv_x         = adv_x
        self.y_target      = y_target
        self.x_probs       = x_probs
        self.adv_x_probs   = adv_x_probs
        self.adv_diff_norm = adv_diff_norm
        self.model_desc    = model_desc


    def show(self):

        fig = plt.figure(figsize=(15, 5))

        diff_img = self.adv_x.clone() - self.x.clone()
        diff_img = diff_img.detach().numpy()
        diff_img = np.abs(diff_img)
        diff_img = diff_img.reshape((28, 28))

        adv_img = self.adv_x.reshape((28, 28))
        adv_img = adv_img.detach().numpy()

        # adversarial image
        fig.add_subplot(1, 3, 1)
        plt.imshow(adv_img, cmap = 'Greys')
        # diff map
        fig.add_subplot(1, 3, 2)
        plt.imshow(diff_img, cmap = 'Greys')
        # probabilities
        fig.add_subplot(1, 3, 3)
        p_range = np.arange(0, 10, 1)
        plt.bar(p_range, self.adv_x_probs)
        plt.show()

    def __str__(self):

        s = ""
        s = s + "Model :  "                  + str(self.model_desc)       + "\n"
        s = s + "x shape               : "   + str(self.x.shape)          + "\n"
        s = s + "adversary shape       : "   + str(self.adv_x.shape)      + "\n"
        s = s + "True label            : "   + str(self.y.item())         + "\n"
        s = s + "Make model think      : "   + str(self.y_target.item())  + "\n"
        s = s + "||x - adversary||2    : "   + str(self.adv_diff_norm)    + "\n"
        s = s + "model probs. with x   : \n" + str(self.x_probs)          + "\n"
        s = s + "model probs. with adv : \n" + str(self.adv_x_probs)      + "\n"

        return s

    def __repr__(self):
        return self.__str__()


## Classifier

class Classifier:

    ## Constructor
    # train_ds : train dataset
    # test_ds  : test dataset
    # batch size : train batch size
    def __init__(self,
                 train_ds,
                 test_ds,
                 batch_size = 100,
                 learning_rate = 0.0005,
                 nepochs = 1000,
                 report_folder = "",
                 desc = "",
                 is_CLF_DS = True): # indicate if the dataset is CLF_ds

        # record inputs in self
        self.train_ds      = train_ds
        self.test_ds       = test_ds
        self.batch_size    = batch_size
        self.Eta           = learning_rate
        self.nepochs       = nepochs
        self.report_folder = report_folder
        self.desc          = desc
        self.is_CLF_DS     = is_CLF_DS

        # initialize epoch
        self.epoch = 0

        # update intervals; in units for epoch
        self.update_interval = 1
        # report gen interval; report every 10 epochs
        self.report_interval = 10

        if is_CLF_DS:
            # print information
            print self.train_ds
            print self.test_ds

            # Sanity check
            assert self.test_ds.D == self.train_ds.D
            assert self.test_ds.L == self.train_ds.L

            # features
            self.D = train_ds.D
            self.L = self.test_ds.L

        else:

            # features
            # example output of shape below ; (1, 28, 28)
            data_shape = train_ds.__getitem__(0)[0].shape
            self.D =  data_shape[1] * data_shape[2]
            self.L = len(set(self.train_ds.train_labels.numpy()))

            print "Train set features : ", self.D
            print "Train set labels # : ", self.L

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

                # reshape data
                data = data.reshape((self.batch_size, self.D))

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


    # eta is the ultimate update factor (learning rate)
    # lam is the regularization parameter in "lambda * || x - x_target ||2^2
    # converge_fn takes 2 arguments. (i) conf. of y_target with adv_x
    #                                (ii) ||x - adv_x||2
    #  When converge function returns true, it is determined that a adversarial
    #  example is found
    def get_adversary(self,
                      x,                # true x
                      y,                # true y
                      y_target,         # make model think adv_x is of class y_target
                      adv_x = None,     # adversarial x; if None, initialize randomly
                      eta = 0.00001,
                      lam = 0.1,
                      converge_fn = None,
                      debug = False):

        assert converge_fn is not None

        image_update_interval = 5000

        if (adv_x is None):
            # random adv_x
            adv_x = torch.tensor(np.random.normal(.5, .3, (1, self.D)))
            adv_x = adv_x.type(torch.FloatTensor)
        else:
            # reshape adv_x properly
            adv_x = adv_x.reshape((1, self.D))
            adv_x = adv_x.type(torch.FloatTensor)


        # reshape x to match adv_x's shape
        x = x.reshape((1, self.D))
        x = x.type(torch.FloatTensor)

        y_target = torch.tensor(y_target.numpy().flatten())

        print "True label ", y
        print "Make network think it is ", y_target

        self.model.eval()

        times = 100000
        time  = 0

        input_diff = 0.0

        initial_x_probs = 0.0

        while time < times:

            # Make adv_x a gradient variable
            adv_x = torch.autograd.Variable(adv_x, requires_grad = True)

            # setup
            self.optimizer.zero_grad()

            # forward
            log_softmax = self.model(adv_x)

            # prediction
            loss = torch.nn.functional.nll_loss(log_softmax, y_target)

            # back prop
            loss.backward()

            #for param in self.model.parameters():
            #    print param.grad
            input_g = adv_x.grad.clone()

            # keep it as close to x as possible
            adv_x = adv_x - eta * (adv_x.grad + lam * (adv_x - x))

            time = time + 1

            # calculate norm of difference between adv_x and x
            diff_v = (adv_x - x).detach().numpy()
            diff_v = diff_v.reshape((self.D))
            diff_norm = l2(diff_v)

            # calculate y_conf and y_target_conf
            y_conf        = round(math.exp(log_softmax[0][y.item()]) * 100, 2)
            y_target_conf = round(math.exp(log_softmax[0][y_target.item()]) * 100, 2)

            if VERBOSE and debug:

                # input gradient norm
                ipg_norm = l2(input_g.numpy().reshape((self.D)))

                verbose_s = "Adv. iteration : " + str(time)             +                                               \
                            " | L2 : "          + str(diff_norm)        +                                               \
                            " | "               + str(y.item())         + " - conf. : " + str(y_conf) + "%"  +          \
                            " | "               + str(y_target.item())  + " - conf. : " + str(y_target_conf) + "%" +    \
                            " | InputG norm : " + str(ipg_norm)         +                                               \
                            " , done. \r"

                sys.stdout.write(verbose_s)
                sys.stdout.flush()


            if debug and (time % image_update_interval == 0 or diff_norm - input_diff > 1.0):

                fig = plt.figure(figsize=(15, 5))
                #fig = plt.figure()

                diff_img = adv_x.clone() - x.clone()
                diff_img = diff_img.detach().numpy()
                diff_img = np.abs(diff_img)
                diff_img = diff_img.reshape((28, 28))

                adv_img = adv_x.reshape((28, 28))
                adv_img = adv_img.detach().numpy()

                # adversarial image
                fig.add_subplot(1, 3, 1)
                plt.imshow(adv_img, cmap = 'Greys')
                # diff map
                fig.add_subplot(1, 3, 2)
                plt.imshow(diff_img, cmap = 'Greys')
                # probabilities
                fig.add_subplot(1, 3, 3)
                p_range = np.arange(0, 10, 1)
                p = np.exp(log_softmax.detach().numpy().reshape((self.L)))
                plt.bar(p_range, p)
                plt.show()

            # update input_diff; it diff_norm is significantly bigger
            if diff_norm - input_diff > 1.0:
                input_diff = diff_norm

            # have we converged yet ?
            if converge_fn(y_target_conf, diff_norm):

                # yes :) but does our adversarial example classify as desired target ?
                if y_target_conf < 50:
                    # no ? :(
                    return None
                else:
                    x_probs_rnd     = np.round(self.probs(x), 2)
                    adv_x_probs_rnd = np.round(self.probs(adv_x), 2)
                    adv_example = AdversarialExample(x = x, y = y,
                                                     adv_x = adv_x,
                                                     y_target= y_target,
                                                     x_probs = x_probs_rnd,
                                                     adv_x_probs = adv_x_probs_rnd,
                                                     adv_diff_norm = diff_norm,
                                                     model_desc = self.desc)
                    return adv_example

    def probs(self, x):

        self.model.eval()

        log_softmax = self.model(x)

        # log softmax to numpy
        p = np.exp(log_softmax.detach().numpy().reshape((self.L)))

        return p

    def test(self, x):

        #assert len(x) == self.train_ds.D

        self.model.eval()

        log_softmax = self.model(x)

        prediction_label = log_softmax[0].max(0)[1]

        return prediction_label

    def train_loss(self):
        return self.loss(self.train_ds)

    def test_loss(self):
        return self.loss(self.test_ds)

    # evaluate loss on the given dataset
    def loss(self, ds):

        loss_ds_batch_size = len(ds)

        dloader = torch.utils.data.DataLoader(ds,
                                              batch_size = loss_ds_batch_size,
                                              shuffle = False)

        acc = 0.0
        l   = 0.0

        ncorrect = 0

        self.model.eval

        with torch.no_grad():

            for data, targets in dloader:

                # reshape data
                data = data.reshape((loss_ds_batch_size, self.D))

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

        train_data_desc = ""
        test_data_desc = ""

        if self.is_CLF_DS:
            train_data_desc = self.train_ds.desc
            test_data_desc  = self.test_ds.desc
        else:
            train_data_desc = "generic"
            test_data_desc  = "generic"

        # create a folder to store current state
        report_folder_now = os.path.join(self.report_folder,
                                         train_data_desc + "_" + \
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

        if self.is_CLF_DS:
            train_data_desc = self.train_ds.desc
            test_data_desc  = self.test_ds.desc
        else:
            train_data_desc = "generic"
            test_data_desc  = "generic"

        # represent self in a string
        s = "========================================================\n"
        s = s + " Model : " + self.desc + "\n"
        s = s + " Dataset Train - " + train_data_desc + "\n"
        s = s + " Dataset Test  - " + test_data_desc  + "\n"
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
                 desc = "Logistic_1L",
                 is_CLF_DS = True):

        # let parent do all the work
        Classifier.__init__(self,
                            train_ds,
                            test_ds,
                            batch_size = batch_size,
                            learning_rate = learning_rate,
                            nepochs = epochs,
                            report_folder = report_folder,
                            desc = desc,
                            is_CLF_DS = is_CLF_DS)

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
                 n_h1 = 0,              # number of nodes in hidden layer
                 report_folder = "",
                 desc = "Logistic_2L",
                 is_CLF_DS = True):

        # let parent do all the work
        Classifier.__init__(self,
                            train_ds,
                            test_ds,
                            batch_size = batch_size,
                            learning_rate = learning_rate,
                            nepochs = epochs,
                            report_folder = report_folder,
                            desc = desc,
                            is_CLF_DS = is_CLF_DS)

        # hidden layer 1 dimensions;
        if n_h1 != 0:
            # hidden layer nodes specified
            self.n_h1 = n_h1
        else:
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
                 n_h1 = 0,
                 report_folder = "",
                 desc = "Logistic_LRL",
                 is_CLF_DS = True):

        # let parent do all the work
        Classifier.__init__(self,
                            train_ds,
                            test_ds,
                            batch_size = batch_size,
                            learning_rate = learning_rate,
                            nepochs = epochs,
                            report_folder = report_folder,
                            desc = desc,
                            is_CLF_DS = is_CLF_DS)

        # hidden layer 1 dimensions;
        if n_h1 != 0:
            # hidden layer nodes specified
            self.n_h1 = n_h1
        else:
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

################################################################################
