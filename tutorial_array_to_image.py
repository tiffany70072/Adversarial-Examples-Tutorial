from cleverhans.utils_mnist import data_mnist
from array_to_image import array_to_image
import numpy as np

X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                train_end=60000,
                                                test_start=0,
                                                test_end=10000)
# visualization for MNIST
array_to_image(X_train[0:5], "mnist")

# visualization for MNIST's adversarial examples
adv_image = np.load("adv_image_FGM.npy")
array_to_image(adv_image[0:5], "adv_image_FGM")

# visualization for Cifar-10's adversarial examples
adv_image = np.load("adv_image_FGM_cifar10.npy")
array_to_image(adv_image[0:5], "adv_image_cifar10", channels = 3, size = 32)
