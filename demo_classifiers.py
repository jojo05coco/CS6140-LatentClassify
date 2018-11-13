## Demo program on how to use the classifiers

from dataset import CLF_DS

from classifiers import Logistic_1L, Logistic_2L, Logistic_LRL

from svm_classifier import SVM_Classifier

# latent codes train path
#latent_codes_train = "./test/AE_MNIST-latent_tr.pth"
#targets_train      = "./test/AE_MNIST-targets_tr.pth"
#latent_codes_test  = "./test/AE_MNIST-latent_ts.pth"
#targets_test       = "./test/AE_MNIST-targets_ts.pth"

latent_codes_train = "./test/AE_FMNIST_latent_tr.pth"
targets_train      = "./test/AE_FMNIST_targets_tr.pth"
latent_codes_test  = "./test/AE_FMNIST_latent_ts.pth"
targets_test       = "./test/AE_FMNIST_targets_ts.pth"

## SVM classifier ##############################################################

svm_clf = SVM_Classifier(latent_codes_train, targets_train, latent_codes_test, targets_test)

svm_clf.train()

exit(-1)

## Neural networks #############################################################

train_ds = CLF_DS(latent_codes_train, targets_train, "FMNIST train dataset")
test_ds  = CLF_DS(latent_codes_test, targets_test, "FMNIST test dataset")

#clf = Logistic_1L(train_ds, test_ds, batch_size = 1000)
#clf = Logistic_2L(train_ds, test_ds, learning_rate = 0.00001, batch_size = 500, epochs = 10000)
clf = Logistic_LRL(train_ds, test_ds, learning_rate = 0.00001, batch_size = 500, epochs = 10000)
clf.train()
