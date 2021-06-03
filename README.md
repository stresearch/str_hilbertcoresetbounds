# STR Hilbert Coreset Bounds

There are some scripts in here that allow for calculation of performance bounds on a given network.
The code uses coresets on a Hilbert space to construct a generalization bound over the data.

## Initialization 

Before running the code, clone the git repo and create a python environment with the required packages.

Run the following command to set up the repo:

```bash
pip install -r requirements.txt
```

Models used for evaluation are provided in the bound evaluation models directory.
We also provide the outputs of the bound computation in the bound calculation directory.
The results are analyzed in our bound analysis notebooks.

# Hilbert Coreset Bound Computation

The code compute generalization bounds on the data using teh Frank-Wolfe algorithm to construct coresets over a Hilbert space.
The primary coreset calculation script takes a model with trained parameters and a dataset on which it operates.
It provides a file with results that contain all necessary parameters for teh computation of the generalization bound.
We provide examples of calculating the generalization bounds for the provided models in the notebooks.

## Usage

You can run any of the provided models to calculate the bounds with Hilbert coresets. 
We have provided two models that can be used for bound computation as well as several checkpoints for each models:
- StrMnist
    - 500
    - 3000
    - 10000
    - 30000
- IsiCifar
    - 500
    - 2500
    - 4000
    - 10000

The coerset calc script has several key parameters:
- model-name: The name of the model
- checkpoint: The checkpoint for the model
- hessian: A location to store the hessian of the loss function
- sample-loc: A location to store the binary loss matrix
- sigma-prior: The standard deviation of the prior
- projection-dimensions: The number of parameter samples we take from the posterior
- num-iterations: The number of iterations to run the coreset construction algorithm
- checkpoint-loc: A location to store periodic checkpoint files
- gpus: The GPU on the machine to use for the computation
- result-loc: Location of file to store the final results of the script
- log-leve: THe level of logging info for the program to provide

Example:

```bash
python3 coreset_calc_script.py --model-name IsiCifar --checkpoint 4000 --hessian bound_calculation/hessian/hess_isi_cifar_4000.npz --sample-loc bound_calculation/samle_isi_cifar_4000/ ----num-iterations 100 500 1000 4000 --checkpoint-loc bound_calculation/checkpoints_isi_cifar_4000/ --gpus 0 --results-loc bound_calculation/isi_cifar_results/ --log-level INFO
```

We also provide a script in example_bound.sh that uses the STR Mnist model as a toy example.
You can try running this to ensure the repo has been installed correctly.
However, this script can still take several hours to complete, as many of the computations are intensive even for small examples.

NOTE: Currently, the computation requires use of a GPU.
This may be changed in the future.

The results will be stored in an npz file with the following attributes:
- iters: The number of Frank-Wolfe algorithm iterations
- coresize: The size of the coresets
- genErr: The expected error of the model
- wminuspnorm_full: The L1 norm of the difference between the coreset weihts and the data distribution
- lnorm_prior: The L2 norm of the risk samples
- llwnorm: The L2 norm of the difference between the loss and coreset of the loss
- lwnorm: The L2 norm of the coreset
- wnorm: The L1 norm of the coreset weights
- params: A list of bound parameters: [sigma, eta, eta_bar, beta, xi]

## Computing bound values

The results file from the coreset calculation script include all necessary parameters to compute the bounds on the model.
For an example, we have computed the bounds on teh STR Mnist model in

```bash
bounds/str_mnist/mnist_bound_computation.ipynb
```

This can be used to compute how the bound changes with number of iterations and coreset size, as well as looking at how the bound depends on the checkpoint.

## Using Your Own Model

The Model module provides an interface between the STR bound calculation code and a model. Documentation for the functions is provided in the module. To calculate bounds for your own in-house model, implement a class that extends teh Model interface. There are 4 key functions to implement:

* get_model_weights: Get the parameters from the model
* set_model_weights: Manually change teh parameters of the model to new values
* model_eval: Evaluate the model on provided data with given weights
* grad_model_out_weights: Get the gradient of the model with respect to the parameters, evaluated at given data points

There are example implementations for the provided models.

You must also add your model as an option to run in the model-name command line argument

## Auxilliary Files

You can see how the STR Mnist model was trained in the provided notebook

```bash
bounds/str_mnist/mnist_training.ipynb
```

The script for bound computation provides files with the information for bound computation saved within them.
The analysis of these results and computation of the actural bound is performed in the notebook:

```bash
bounds/str_mnist/mnist_bound_computation.ipynb
```

Similarly, we analyze the results for the ISI Cifar model in

```bash
bounds/isi_cifar/isi_cifar_results.ipynb
```
