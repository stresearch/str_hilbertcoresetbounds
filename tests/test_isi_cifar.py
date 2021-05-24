"""
Copyright 2020-2021 Systems & Technology Research

This module is a series of pytest test functions for the ISI Cifar model interface in models/model_ta1_isi_cifar.py
"""
import pytest
import numpy as np
from torch.utils.data import Subset
from bounds.models.model_ta1_isi_cifar import IsiCifar


@pytest.mark.quick
@pytest.mark.parametrize("checkpoint", ["500", "2500", "4000", "10000"])
def test_init(checkpoint):
    model = IsiCifar(checkpoint, "../../data")
    assert model.mdl is not None
    assert model.labels.size == 50000
    assert model.train_labels.size == int(checkpoint)


@pytest.mark.quick
@pytest.mark.parametrize("checkpoint", ["500", "2500", "4000", "10000"])
def test_get_weights(checkpoint):
    model = IsiCifar(checkpoint, "../../data")
    weights = model.get_model_weights()
    assert weights.size == 3131364


@pytest.mark.quick
@pytest.mark.parametrize("checkpoint", ["500", "2500", "4000", "10000"])
def test_set_weights(checkpoint):
    model = IsiCifar(checkpoint, "../../data")
    weights = model.get_model_weights()
    new_weights = np.repeat(1, 3131364)
    model.set_model_weights(new_weights)
    assert np.max(np.abs(model.get_model_weights() - new_weights)) < 1e-4


@pytest.mark.parametrize("checkpoint", ["500", "2500", "4000", "10000"])
def test_model_eval(checkpoint):
    model = IsiCifar(checkpoint, "../../data")
    opt_weights = model.get_model_weights()
    out = model.model_eval(opt_weights, model.train_set)
    pred = np.argmax(out, axis=1)
    assert np.array_equal(pred, model.train_labels)


@pytest.mark.quick
@pytest.mark.parametrize("checkpoint", ["500", "2500", "4000", "10000"])
def test_zero_one_loss(checkpoint):
    model = IsiCifar(checkpoint, "../../data")
    opt_weights = model.get_model_weights()
    loss_val = model.loss_val_zero_one(model.train_set, model.train_labels, opt_weights)
    loss = np.zeros(model.train_labels.size)
    assert np.array_equal(loss, loss_val)


@pytest.mark.parametrize("checkpoint", ["500", "2500", "4000", "10000"])
def test_grad_model(checkpoint):
    model = IsiCifar(checkpoint, "../../data")
    opt_weights = model.get_model_weights()
    grad_data = Subset(model.dataset, np.array([0, 1]))
    grad, out = model.grad_model_out_weights(grad_data, opt_weights)
    out_vals = model.model_eval(opt_weights, grad_data)
    assert np.max(np.abs(out - out_vals)) < 1e-4
    assert grad.shape == (2, opt_weights.size)


@pytest.mark.quick
@pytest.mark.parametrize("checkpoint", ["500", "2500", "4000", "10000"])
def test_hess_diag(checkpoint):
    model = IsiCifar(checkpoint, "../../data")
    opt_weights = model.get_model_weights()
    grad_data = Subset(model.dataset, np.array([0, 1]))
    true_labels = model.labels[:2]
    hess_diag = model.g_diag(grad_data, true_labels, opt_weights)
    assert hess_diag.size == opt_weights.size
    assert np.all(hess_diag >= 0)
