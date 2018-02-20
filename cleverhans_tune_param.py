from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper

FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def cnn_model_BIM(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=32, nb_classes=10):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)
    
    layers = [Conv2D(nb_filters, (3, 3), strides = (1, 1), padding="same", input_shape=input_shape),
            Activation('relu'),
            Conv2D(nb_filters, (3, 3), strides = (1, 1), padding="valid"),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(nb_filters*2, (3, 3), strides = (1, 1), padding="valid"),
            Activation('relu'),
            Conv2D(nb_filters*2, (3, 3), strides = (1, 1), padding="valid"),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(nb_classes)]

    for layer in layers: model.add(layer)

    if logits: logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits: return model, logits_tensor
    else: return model

def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128, epsilon = 0.3,
                   learning_rate=0.001, train_dir="/tmp",
                   filename="mnist.ckpt", load_model=False,
                   testing=False):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param train_dir: Directory storing the saved model
    :param filename: Filename to save model under
    :param load_model: True for load, False for not load
    :param testing: if true, test error is calculated
    :return: an AccuracyReport object
    """
    keras.layers.core.K.set_learning_phase(0)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = cnn_model_BIM()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': train_dir,
        'filename': filename
    }
    ckpt = tf.train.get_checkpoint_state(train_dir)
    ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path

    rng = np.random.RandomState([2017, 8, 30])
    if load_model and ckpt_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        print("Model loaded from: {}".format(ckpt_path))
        evaluate()
    else:
        print("Model was not loaded, training from scratch.")
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, save=False, rng=rng)

    # Calculate training error
    if testing:
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_train, Y_train, args=eval_params)
        report.train_clean_train_clean_eval = acc

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    wrap = KerasModelWrapper(model)

    print("FastGradientMethod")
    fgsm1 = FastGradientMethod(wrap, sess=sess)
    for epsilon in [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        print("Epsilon =", epsilon),
        fgsm_params = {'eps': epsilon,
                       'clip_min': None,
                       'clip_max': None}
        adv_x = fgsm1.generate(x, **fgsm_params)
        # Consider the attack to be constant
        adv_x = tf.stop_gradient(adv_x)
        preds_adv = model(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc

    print("BasicIterativeMethod")
    bim = BasicIterativeMethod(wrap, sess=sess)
    for epsilon, order in zip([0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 0.5, 1.0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2, 2]):
        print("Epsilon =", epsilon),
        fgsm_params = {'eps': epsilon,
                       'clip_min': 0.,
                       'clip_max': 1., 
                       'ord': order}
        adv_x = bim.generate(x, **fgsm_params)
        # Consider the attack to be constant
        adv_x = tf.stop_gradient(adv_x)
        preds_adv = model(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc

    # Calculating train error
    if testing:
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_train,
                         Y_train, args=eval_par)
        report.train_clean_train_adv_eval = acc
    return

    print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = cnn_model()
    preds_2 = model_2(x)
    wrap_2 = KerasModelWrapper(model_2)
    #fgsm2 = FastGradientMethod(wrap_2, sess=sess)
    bim2 = BasicIterativeMethod(wrap_2, sess=sess)
    preds_2_adv = model_2(bim2.generate(x, **fgsm_params))

    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy

    # Perform and evaluate adversarial training
    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, save=False, rng=rng)

    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_train, Y_train,
                              args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, preds_2_adv, X_train,
                              Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracy

    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs,
                       batch_size=FLAGS.batch_size,
                       learning_rate=FLAGS.learning_rate,
                       train_dir=FLAGS.train_dir,
                       filename=FLAGS.filename,
                       load_model=FLAGS.load_model, 
                       epsilon=epsilon)

if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('train_dir', 'tmp', 'Directory where to save model.')
    flags.DEFINE_string('filename', 'mnist_BasicIterative_.ckpt', 'Checkpoint filename.')
    flags.DEFINE_boolean('load_model', True, 'Load saved model or train.')
    tf.app.run()

