
import pickle

from classifiers import AdversarialExample

import numpy as np

#adversary_path = "./mnist_img_advex/Logistic_1L_advex/advex.pkl"
#adversary_path = "./mnist_img_advex/Logistic_LRL_advex/advex.pkl"
#adversary_path = "./adversary_dataset/CONFIDENCE_50/fmnist_img_50conf_advex/Logistic_LRL_advex/advex.pkl"

adversary_path = "./MNIST_HD_advex/Logistic_2L_advex/advex.pkl"

af = open(adversary_path, "rb")
exs = pickle.load(af)
af.close()

# remove all adversary that is None
exs = filter(lambda ex: ex is not None, exs)
# remove all adversary whose norm is > 1.0
#exs = filter(lambda ex: ex.adv_diff_norm < 1.0, exs)
# sort examples by increasing diff norm
#exs.sort(key = lambda ex: ex.adv_diff_norm)

print "Examples ", len(exs)

for ex in exs:

    print ex
    ex.show()

    # max and min values in img_ex's adversary
    adv_x_max = np.max(ex.adv_x.detach().numpy()[0])
    adv_x_min = np.min(ex.adv_x.detach().numpy()[0])

    assert adv_x_min >= 0.0 and adv_x_max <= 1.0
