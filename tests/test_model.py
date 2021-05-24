"""
Copyright 2020-2021 Systems & TEchnology Research

This module is a series of pytest tests of the model interface in models/__init__.py
"""
from bounds.models import Model
import pytest
import numpy as np


@pytest.mark.quick
def test_zero_one_loss():
    y_true = np.array([2, 1])
    y_pred = np.array([[0.98351, 0.12042, 0.14253], [0.01924, 0.99032, 0.01582]])
    loss_val = Model.point_zero_one_loss(y_pred, y_true)
    loss = np.array([1, 0])
    assert np.array_equal(loss, loss_val)


@pytest.mark.quick
def test_cross_entropy_loss():
    y_true = np.array([2, 1])
    y_pred = np.array([[0.98351, 0.12042, 0.14253], [0.01924, 0.99032, 0.01582]])
    loss_val = Model.point_crossentropy_loss(y_pred, y_true)
    loss = np.array([1.948202, 0.0166274])
    assert abs(np.max(loss_val - loss)) < 1e-6


@pytest.mark.quick
def test_cross_entropy_deriv():
    y_true = np.array([2, 1])
    y_pred = np.array([[0.98351, 0.12042, 0.14253], [0.01924, 0.99032, 0.01582]])
    loss_val = Model.deriv_2_loss_model_val(y_pred, y_true)
    loss = np.array([49.2251932, 1.0196447])
    assert abs(np.max(loss_val - loss)) < 1e-6
