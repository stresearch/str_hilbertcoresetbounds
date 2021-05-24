"""
Copyright 2020-2021 Systems & Technology Research

This module samples network weights from teh posterior distribution and calculates teh binary loss matrix for the network
"""
import numpy as np
import cupy as cp
import os
import math
from bayesiancoresets.util.norm_calc import NormCalc
import bayesiancoresets as bc
import logging

log = logging.getLogger("str_lwll")


def sample_posterior(
    mu_post, stdev_post, num_samples, loss, sample_loc, gpus, data_size
):
    avail = NormCalc.memory_avail()
    avail = avail / 4
    max_length = math.floor(avail / (8 * mu_post.shape[0]))
    num_chunks = math.ceil(num_samples / max_length)
    log.info(
        f"Taking {num_samples} samples from teh posterior in batches of {max_length}"
    )

    def sampler(sz, w=None, ids=None):
        with cp.cuda.Device(gpus[0]):
            theta = cp.random.normal(0, 1, (sz, mu_post.shape[0]))
            theta = theta * cp.asarray(stdev_post) + cp.asarray(mu_post)
            return theta.get()

    if not os.path.exists(sample_loc):
        log.info(f"Creating location to store samples at {sample_loc}")
        os.mkdir(sample_loc)

    file_loc = []
    for i in range(num_chunks):
        if i == (num_chunks - 1):
            length = num_samples % max_length
        else:
            length = max_length
        tsf = bc.BayesianTangentSpaceFactory(loss, sampler, length)
        sample = tsf()
        sample = np.swapaxes(sample, 0, 1)
        file_loc.append(sample_loc + "/sample_" + str(i) + ".npy")
        np.save(file_loc[i], sample)
        if i % 50 == 0:
            log.info(f"completed {i} batches of sampling from posterior")

    log.info(f"Finished sampling from posterior, constructing data matrix")
    return NormCalc(file_loc, num_samples, data_size, gpus)
