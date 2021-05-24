"""
Copyright Systems & Technology Research 2020-2021

This module calculates the coreset for the bound computation
"""
import numpy as np
import bayesiancoresets as bc
from os import path
import math
import logging

log = logging.getLogger("str_lwll")


# Load dataset
algdict = {
    "FW": bc.snnls.FrankWolfe,
}


def calc_coreset(
    iterations,
    likelihood_matrix,
    probability,
    checkpoint_its,
    checkpoint_loc,
    temp_file_loc,
):
    coreset_sizes = np.zeros(iterations.shape[0])
    norm_difference = np.zeros(iterations.shape[0])
    norm_coreset = np.zeros(iterations.shape[0])
    norm_loss = np.zeros(iterations.shape[0])
    w_minus_p_norm = np.zeros(iterations.shape[0])
    w_norm = np.zeros(iterations.shape[0])
    gen_err = np.zeros(iterations.shape[0])
    log.info(f"Initialized coreset calculation for {iterations} iterations")

    scale_loc = []
    for file in likelihood_matrix.files:
        base_file = path.basename(file)
        scale_loc.append(temp_file_loc + base_file)
    alg = bc.HilbertCoreset(
        likelihood_matrix,
        probability,
        scale_loc,
        checkpoint_its,
        checkpoint_loc,
        likelihood_matrix.gpu_list,
        snnls=algdict["FW"],
    )

    for i in range(len(iterations)):
        if i != 0:
            iterate = iterations[i] - iterations[i - 1]
        else:
            iterate = iterations[0]
        size = iterations[i]

        coreset_vals = build_coreset(
            likelihood_matrix, iterate, size, probability, alg, checkpoint_loc
        )
        log.info(f"Completed {size} iterations of FW algorithm")
        log.info(f"Current size {coreset_vals[0]}")
        coreset_sizes[i] = coreset_vals[0]
        log.info(f"Norm of difference between coreset and full set: {coreset_vals[1]}")
        norm_difference[i] = coreset_vals[1]
        log.info(f"Norm of coreset: {coreset_vals[3]}")
        norm_coreset[i] = coreset_vals[3]
        log.info(f"Norm of full set {coreset_vals[2]}")
        norm_loss[i] = coreset_vals[2]
        log.info(
            f"L1 norm of difference between weights and probability: {coreset_vals[4]}"
        )
        w_minus_p_norm[i] = coreset_vals[4]
        log.info(f"L1 norm of coreset weights: {coreset_vals[5]}")
        w_norm[i] = coreset_vals[5]
        log.info(f"Generalization error: {coreset_vals[6]}")
        gen_err[i] = coreset_vals[6]

    return (
        coreset_sizes,
        norm_difference,
        norm_coreset,
        norm_loss,
        w_minus_p_norm,
        w_norm,
        gen_err,
    )


def build_coreset(likelihood, iterations, size, probability, alg, checkpoint_loc):
    # this runs alg up to a level of M; on the next iteration, it will continue from where it left off
    alg.build(iterations, size)
    wts, idcs = alg.weights()

    csizes_train = wts.shape[0]
    weights = np.zeros(likelihood.num_cols)
    weights[idcs] = wts
    weights_tilde = weights * probability
    gfs = likelihood.dot_product(probability)
    gcs = likelihood.dot_product(weights_tilde)
    log.debug(f"Number of non-zero weights: {wts.shape}")
    norm_difference = math.sqrt(np.mean(((gcs - gfs) ** 2)))
    log.debug(f"Distance between coreset and full: {norm_difference}")
    norm_loss = math.sqrt(np.mean((gfs ** 2)))
    norm_coreset = math.sqrt(np.mean((gcs ** 2)))
    log.debug(f"Norm: {norm_loss}")
    log.debug(f"core: {norm_coreset}")
    w_minus_p_norm = np.sum(np.abs(weights_tilde - probability))
    log.debug(f"w nminus p norn: {w_minus_p_norm}")
    w_norm = np.sum(np.abs(weights_tilde))
    log.debug(f"w norm: {w_norm}")
    gen_err = np.mean(np.abs(gfs))

    checkpoint_vec = np.array(
        [
            csizes_train,
            norm_difference,
            norm_loss,
            norm_coreset,
            w_minus_p_norm,
            w_norm,
            gen_err,
        ]
    )

    np.savez(
        checkpoint_loc + "checkpoint_" + str(size) + ".npz",
        checkpoint_Vals=checkpoint_vec,
        w=weights_tilde,
    )

    return (
        csizes_train,
        norm_difference,
        norm_loss,
        norm_coreset,
        w_minus_p_norm,
        w_norm,
        gen_err,
    )
