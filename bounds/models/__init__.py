"""
Copyright 2020-2021 Systems & Technology Research

This module is an interface between the model and the bound computation
"""
import numpy as np
import logging
import os
from abc import abstractmethod, ABC

DIR_NAME = os.path.dirname(os.path.realpath(__file__)).split("/bound")[0] + "/"


log = logging.getLogger("str_lwll")


class Model(ABC):
    """Abstract base class for running the modelss
       In order to run your own model, create a subclass
       that implements the abstract methods in this API
       State variables:
       self.mdl                                     Model to evaluate
       self.dataset      <torchvision.datasets>     Full data set
       self.labels       <numpy.array>              Labels for data set
       self.train_set    <torch.utils.data.Subset>  Labeled data for training
       self.train_labels <numpy.array>              Labels used for training"""

    @classmethod
    def weights_from_v_g(cls, weight_v, weight_g):
        """Convert from weight_v and weight_g to weight"""
        nrm = np.sqrt(
            np.sum(
                np.power(
                    weight_v.reshape(
                        [weight_v.shape[0], -1] + [1] * (len(weight_v.shape) - 2)
                    ),
                    2,
                ),
                axis=1,
                keepdims=True,
            )
        )
        return weight_g * weight_v / nrm

    @abstractmethod
    def get_model_weights(self):
        """Get the current weights for the model
           return <numpy.array> model weights and biases"""

    @abstractmethod
    def set_model_weights(self, params):
        """Changes teh weights in the model
           params <numpy.array> new model weights and biases"""

    @abstractmethod
    def model_eval(self, params, dataset):
        """Runs inference on teh model for the given input
           params  <numpy.array>             model weights and biases
           dataset <torch.utils.data.Subset> Data to evaluate
           return  <numpy.array>             model predictions"""

    @classmethod
    def point_crossentropy_loss(cls, y_pred, y_true):
        """Calculates the cross-entropy loss
           y_pred <numpy.array> predicted labels
           y_true <numpy.array> true labels"""
        y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)
        indicator = np.zeros(y_pred.shape)
        for i in range(indicator.shape[0]):
            indicator[i, y_true[i]] = 1
        loss = np.sum(-1 * indicator * np.log(y_pred), axis=1)
        return loss

    @classmethod
    def point_zero_one_loss(cls, y_pred, y_true):
        """Calculates teh 0-1 loss
           y_pred <numpy.array> predicted labels
           y_true <numpy.array> true labels"""
        pred_idx = np.argmax(y_pred, axis=1)
        return (pred_idx != y_true).astype(np.int8)

    def loss_val(self, dataset, y_true, params):
        """Calculates cross-entropy loss of the model
           dataset <torch.utils.data.Subset>  Data to evaluate
           y_true  <numpy.array>              True labels
           params  <numpy.array>              weights and biases for the model
           return  <numpy.array>               Loss at the data points"""
        y_pred = self.model_eval(params, dataset)
        return Model.point_crossentropy_loss(y_pred, y_true)

    def loss_val_zero_one(self, dataset, y_true, params):
        """Calculates the 0-1 loss of the model
           Calculates cross-entropy loss of the model
           dataset <torch.utils.data.Subset>  Data to evaluate
           y_true  <numpy.array>              True labels
           params  <numpy.array>              weights and biases for the model
           return  <numpy.array>              Loss at the data points"""
        y_pred = self.model_eval(params, dataset)
        return Model.point_zero_one_loss(y_pred, y_true)

    @abstractmethod
    def grad_model_out_weights(self, datasest, params):
        """Calculate the Gradient of the model wrt the given parameters
           dataset <torch.utils.data.Subst>       Data to eevaluate the gradient at
           params  <numpy.array>                  Model weights and biases for teh derivative
           return (<numpy.array>, <numpy.array>)  Returns 2 arrays
           return[0]  <numpy.array>               2D array of derivatives
           return[1] <numpy.array>                Array of model outputs"""

    @classmethod
    def deriv_2_loss_model_val(cls, y_pred, y_true):
        """Calculate the 2nd derivative of cross-entropy loss wrt model output
           y_true <numpy.array>             the true labels for the data
           y_pred <numpy.array>             the predicted labels for teh array
           return <numpy.array>             The 2nd derivative of cross-entropy loss"""
        y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)
        indicator = np.zeros(y_pred.shape)
        y_true = y_true.astype(int)
        for i in range(indicator.shape[0]):
            indicator[i, y_true[i]] = 1
        hess = np.sum(indicator / y_pred ** 2, axis=1)
        return hess

    def g_diag(self, dataset, y_true, params):
        """Calculate the diagonal of the Hessian of teh cross-entropy loss wrt optimal weights
           dataset  <torch.utils.data.Subset> The dataset to take the derivative
           y_ture   <numpy.array>             The true labels for the data
           params   <numpy.array>             the optimal weights of the model
           return   <numpy.array>             The diagonal of the Hessian
           Note: If the provided parameters are sub-optimal, the hessian calculation is not valid"""
        log.info("Start computing the diagonal of the Hessian")
        hess, y_pred = self.grad_model_out_weights(dataset, params)
        log.debug(f"Got gradient from model")
        for i in range(hess.shape[0]):
            hess[i, :] = hess[i, :] ** 2
        log.debug(f"squared gradien, shape {hess.shape}")
        deriv = Model.deriv_2_loss_model_val(y_pred, y_true)
        log.debug(f"Got loss derivative shape {deriv.shape}")
        hess_diag = np.zeros(hess.shape[1])
        for i in range(hess.shape[1]):
            hess_diag[i] = np.matmul(deriv, hess[:, i]) / y_true.shape[0]
        log.info(
            "Finished computing the diagonal of teh Hessian shape {hess_diag.shape}"
        )
        return hess_diag
