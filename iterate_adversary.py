
import pickle

from classifiers import AdversarialExample

#adversary_path = "./mnist_img_advex/Logistic_1L_advex/advex.pkl"
#adversary_path = "./mnist_img_advex/Logistic_LRL_advex/advex.pkl"
adversary_path = "./adversary_dataset/CONFIDENCE_50/mnist_img_advex/Logistic_1L_advex/advex.pkl"

af = open(adversary_path, "rb")
exs = pickle.load(af)
af.close()

# remove all adversary that is None
exs = filter(lambda ex: ex is not None, exs)
# remove all adversary whose norm is > 1.0
exs = filter(lambda ex: ex.adv_diff_norm < 1.0, exs)

for ex in exs:
    print ex
    ex.show()
    c = raw_input("Press enter to see next adversary ")
