import os
import numpy as np
from bounds.common.post_sample import sample_posterior
from bounds.common.coreset_cal import calc_coreset
from bounds.common.utils import get_bound_params
from bounds.models import DIR_NAME
import logging

log = logging.getLogger("str_lwll")


def main(
    checkpoint,
    hessian_loc,
    sample_loc,
    iters,
    checkpoint_loc,
    sigma_prior,
    projection_dim,
    gpus,
    results_loc,
):
    if not os.path.exists(sample_loc):
        log.info(f"Creating sample save location at {sample_loc}")
        os.mkdir(sample_loc)

    if not os.path.exists(checkpoint_loc):
        log.info(f"Creating checkpoint save location at {checkpoint_loc}")
        os.mkdir(checkpoint_loc)

    log.info(f"Creating model for checkpoint {checkpoint} and getting optimal weights")
    model = Model(checkpoint)
    opt_weights = model.get_model_weights()

    if os.path.exists(hessian_loc):
        log.info(f"Loading the saved hessian at {hessian_loc}")
        hess = np.load(hessian_loc)
    else:
        log.info(f"No hessian found, calculating the hessian")
        hess = model.g_diag(model.train_set, model.train_labels, opt_weights)
        np.save(hessian_loc, hess)
        log.info(f"Hessian saved to {hessian_loc}")

    sigma_post = find_posterior(sigma_prior, hess)

    def loss(prms):
        log.debug(f"Params shape {prms.shape}")
        loss = np.hstack(
            [
                model.loss_val_zero_one(model.dataset, model.labels, prms[i, :])[
                    :, np.newaxis
                ]
                for i in range(prms.shape[0])
            ]
        )
        loss = np.reshape(loss, (loss.shape[0], loss.shape[1], 1))
        log.debug(f"Loss shape {loss.shape}")
        return loss

    log.info("Starting to sample loss from the posterior")
    post_sample = sample_posterior(
        opt_weights,
        sigma_post,
        projection_dim,
        loss,
        sample_loc,
        gpus,
        model.labels.size,
    )
    log.info("Finished sampling loss from the posterior")

    prob = np.repeat(1 / model.labels.size, model.labels.size)
    temp_loc = sample_loc + "/temp"
    if not os.path.exists(temp_loc):
        os.mkdir(temp_loc)

    log.info("Starting coreset calculation with Frank-Wolfe algorithm")
    checkpoint_its = 50
    (
        coreset_sizes,
        norm_difference_full,
        norm_coreset_full,
        norm_loss_full,
        w_minus_p_norm_full,
        w_norm_full,
        gen_err_full,
    ) = calc_coreset(iters, post_sample, prob, checkpoint_its, checkpoint_loc, temp_loc)

    log.info("Starting computing constants for bound")
    params = get_bound_params(post_sample, prob)

    log.info(f"Coreset calculation completed, saving results to {results_loc}")
    np.savez(
        results_loc,
        iters=iters,
        coresize=coreset_sizes,
        genErr=gen_err_full,
        wminuspnorm_full=w_minus_p_norm_full,
        lnorm_prior=norm_loss_full,
        llwnorm=norm_difference_full,
        lwnorm=norm_coreset_full,
        wnorm=w_norm_full,
        params=params,
    )


def find_posterior(sigma, hess_diag):
    assert np.all(hess_diag >= 0)

    idc = np.argwhere(hess_diag < 1 / sigma ** 2)
    cov_post_inv = hess_diag
    cov_post_inv[idc] = 1 / sigma ** 2
    cov_post = 1 / cov_post_inv
    std_post = np.sqrt(cov_post)

    return std_post


if __name__ == "__main__":
    import argparse

    # Define command line arguments
    parser = argparse.ArgumentParser(
        description="Main pipeline for the Hilbert coreset calculation"
    )
    parser.add_argument(
        "--model-name", "-m", type=str, help="The TA1 model/dataset being used"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        help="The checkpoint of the model being evaluated",
    )
    parser.add_argument(
        "--hessian",
        "-e",
        type=str,
        help="npz file location containing the Hessian. "
        + "If the file exists, it will be used as the Hessian. "
        + "If it does not exist, the hessian will be computed and saved here",
    )
    parser.add_argument(
        "--sample-loc",
        "-l",
        type=str,
        help="The location of the samples of the loss from the posterior",
    )

    parser.add_argument(
        "--sigma-prior",
        "-s",
        default=0.001,
        type=float,
        help="The prior standard deviation",
    )
    parser.add_argument(
        "--projection-dimensions",
        "-j",
        default=7000,
        type=int,
        help="The number of samples from the posterior",
    )
    parser.add_argument(
        "--num-iterations",
        "-n",
        type=int,
        nargs="+",
        help="[List] THe number of Frank-Wolfe iterations for bound computation",
    )
    parser.add_argument(
        "--checkpoint-loc",
        "-k",
        type=str,
        help="THe location to store the checkpoints for the bound computation",
    )
    parser.add_argument(
        "--gpus", "-g", type=int, nargs="+", help="List of available GPU (ints)s"
    )
    parser.add_argument(
        "--results-loc",
        "-r",
        type=str,
        help="Location to save the results of the computation",
    )

    parser.add_argument(
        "--log-level",
        "-o",
        default="Info",
        type=str,
        help="The level of the logging info",
    )

    args = parser.parse_args()

    # create logger with 'str_lwll' as logger name
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler("str_lwll.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    if args.log_level.lower() == "info":
        log.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif args.log_level.lower() == "debug":
        log.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif args.log_level.lower() == "warning":
        log.setLevel(logging.WARNING)
        fh.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    elif args.log_level.lower() == "error":
        log.setLevel(logging.ERROR)
        fh.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)

    log_format = "[{asctime}][{filename}][line {lineno}] : {message}"
    formatter = logging.Formatter(log_format, style="{")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    # Define the model we are using and the iterations

    if args.model_name.lower() == "isicifar":
        log.info(f"Model {args.model_name} selected: importing IsiCifar")
        from bounds.models.model_ta1_isi_cifar import IsiCifar as Model
    elif args.model_name.lower() == "strmnist":
        log.info(f"Model {args.model_name} selected: importing StrMnist")
        from bounds.models.model_test_str_mnist import StrMnist as Model
    else:
        log.error(
            f"Model {args.model_name} not implemented, please implement a Model object for this model"
        )
        raise Exception(f"{args.model_name} is an invalid argument")

    iters = np.array(args.num_iterations)

    # Run coreset script
    main(
        args.checkpoint,
        args.hessian,
        args.sample_loc,
        iters,
        args.checkpoint_loc,
        args.sigma_prior,
        args.projection_dimensions,
        args.gpus,
        args.results_loc,
    )
