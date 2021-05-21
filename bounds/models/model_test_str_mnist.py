import numpy as np
from keras import models, losses, datasets
import keras.backend as K
from bounds.models import Model, DIR_NAME

import sys
import logging


log = logging.getLogger("str_lwll")


def logistic_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, -9, 9)
    loss = K.log(1 + K.exp(-1 * y_true * y_pred))
    return K.mean(loss) / K.log(2.0)


class StrMnist(Model):
    """A simple neural net that does Binary Mnist classification"""

    def __init__(self, checkpoint):
        log.info("Loading STR Binary Mnist model")
        losses.logistic_loss = logistic_loss
        self.mdl = models.load_model(
            DIR_NAME
            + "/bound_evaluation_models/str_binary_mnist/binary_mnist_"
            + checkpoint
            + ".h5"
        )
        log.info("Sucessfully loaded model")

        log.info("Loading binary mnist data")
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train / 255
        x_test = x_test / 255
        label_train = np.repeat(0, y_train.shape)
        ind = np.argwhere(y_train > 4)
        label_train[ind] = 1
        label_test = np.repeat(0, y_test.shape)
        ind = np.argwhere(y_test > 4)
        label_test[ind] = 1

        sample = np.load(
            DIR_NAME + "bound_evaluation_models/str_binary_mnist/binary_mnist_samples.npy"
        )
        self.train_set = x_train[sample[: int(checkpoint)]]
        self.train_labels = label_train[sample[: int(checkpoint)]]

        self.dataset = np.concatenate((x_train, x_test), axis=0)
        self.labels = np.concatenate((label_train, label_test))

    def model_eval(self, params, x):
        self.set_model_weights(params)
        out = self.mdl.predict(x).flatten()
        out = ((1 + np.sign(out)) / 2).astype(int)
        one_hot = np.zeros((out.size, 2))
        for i in range(out.size):
            one_hot[i, out[i]] = 1
        return one_hot

    def get_model_weights(self):
        weights = np.array([])

        for layer in self.mdl.layers:
            for model_weights in layer.get_weights():
                weights = np.concatenate((weights, model_weights.flatten()))

        return weights

    def set_model_weights(self, params):
        n = 0

        for layer in self.mdl.layers:
            old_weights = layer.get_weights()
            new_weights = [*range(len(old_weights))]
            for i in range(len(old_weights)):
                weights = params[n : n + old_weights[i].size]
                weights = np.reshape(weights, old_weights[i].shape)
                new_weights[i] = weights
                n = n + old_weights[i].size
            layer.set_weights(new_weights)

        return

    def grad_model_out_weights(self, dataset, params):
        output = self.model_eval(params, dataset)

        self.set_model_weights(params)
        output_tensor = self.mdl.output
        variable_tensor_list = self.mdl.trainable_weights
        gradients = K.gradients(output_tensor, variable_tensor_list)
        sess = K.get_session()
        log.debug("Creating matrix")
        model_grad = np.zeros((dataset.shape[0], params.size), dtype=np.float32)
        log.debug(f"Created matrix, performing {dataset.shape[0]} iterations")
        for i in range(dataset.shape[0]):
            x_in = dataset[i, :]
            x_in = np.reshape(x_in, (1, 784))
            evaluated_gradients = sess.run(gradients, feed_dict={self.mdl.input: x_in})

            grad = np.array([])
            for gradient in evaluated_gradients:
                grad = np.concatenate((grad, gradient.flatten()))
            model_grad[i, :] = grad

        return model_grad, output
