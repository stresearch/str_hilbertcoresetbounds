"""
Copyright 2020-2021 Systems & Technology Research

This module is a series of pytest tests for the STR Mnist model interface at models/model_test_str_mnits.py
"""
import pytest
import numpy as np
from bounds.models.model_test_str_mnist import StrMnist


@pytest.mark.quick
@pytest.mark.parametrize("checkpoint", ["500", "3000", "10000", "30000"])
def test_init(checkpoint):
    model = StrMnist(checkpoint)
    assert model.mdl is not None
    assert model.labels.size == 70000
    assert model.train_labels.size == int(checkpoint)


@pytest.mark.quick
@pytest.mark.parametrize("checkpoint", ["500", "3000", "10000", "30000"])
def test_get_weights(checkpoint):
    model = StrMnist(checkpoint)
    weights = model.get_model_weights()
    assert weights.size == 1192801


@pytest.mark.quick
@pytest.mark.parametrize("checkpoint", ["500", "3000", "10000", "30000"])
def test_set_weights(checkpoint):
    model = StrMnist(checkpoint)
    new_weights = np.repeat(1, 1192801)
    model.set_model_weights(new_weights)
    assert np.max(np.abs(model.get_model_weights() - new_weights)) < 1e-4


@pytest.mark.parametrize(
    "checkpoint, predictions",
    [
        ("500", np.array([[0, 1], [0, 1]])),
        ("3000", np.array([[0, 1], [1, 0]])),
        ("10000", np.array([[0, 1], [1, 0]])),
        ("30000", np.array([[0, 1], [1, 0]])),
    ],
)
def test_model_eval(checkpoint, predictions):
    model = StrMnist(checkpoint)
    opt_weights = model.get_model_weights()
    test_data = model.dataset[:2, :]
    out = model.model_eval(opt_weights, test_data)
    assert np.array_equal(out, predictions)


@pytest.mark.quick
@pytest.mark.parametrize(
    "checkpoint, loss",
    [
        ("500", np.array([0, 1])),
        ("3000", np.array([0, 0])),
        ("10000", np.array([0, 0])),
        ("30000", np.array([0, 0])),
    ],
)
def test_zero_one_loss(checkpoint, loss):
    model = StrMnist(checkpoint)
    opt_weights = model.get_model_weights()
    test_data = model.dataset[:2, :]
    loss_val = model.loss_val_zero_one(test_data, model.labels[:2], opt_weights)
    assert np.array_equal(loss, loss_val)


@pytest.mark.parametrize("checkpoint", ["500", "3000", "10000", "30000"])
def test_grad_model(checkpoint):
    model = StrMnist(checkpoint)
    opt_weights = model.get_model_weights()
    test_data = model.dataset[:2, :]
    grad, out = model.grad_model_out_weights(test_data, opt_weights)
    out_vals = model.model_eval(opt_weights, test_data)
    assert np.max(np.abs(out - out_vals)) < 1e-4
    assert grad.shape == (2, opt_weights.size)


@pytest.mark.quick
@pytest.mark.parametrize("checkpoint", ["500", "3000", "10000", "30000"])
def test_hess_diag(checkpoint):
    model = StrMnist(checkpoint)
    opt_weights = model.get_model_weights()
    test_data = model.dataset[:2, :]
    true_labels = model.labels[:2]
    hess_diag = model.g_diag(test_data, true_labels, opt_weights)
    assert hess_diag.size == opt_weights.size
    assert np.all(hess_diag >= 0)
