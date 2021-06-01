# STR Hilbert Coreset Bounds

There are some scripts in here that allow for calculation of performance bounds on a given network.
The code uses coresets on a hilbert space to construct a generalization bound over the data.

## Initialization 

Before running the code, clone the git repo and create a python environment with the required packages.

Run the following command to set up the repo:

```bash
pip install -r requirements.txt
```

Models used for evaluation are provided in the boundd evaluation models directory.
We also provide the outputs of the bound computation in the bound calculation directory.
The results are analyzed in our bound analysis notebooks.

# Hilbert Coreset Bound Computation

These bounds are based on the Hilbert Coreset formulation in [this] pape (link wiill be added in future). We provide teh code to run teh Hilbert coreset calculations used in that paper.

## Usage

You can run any of the provided models to calculate the bounds with Hilbert coresets. 

Example:

```bash
python3 coreset_calc_script.py --model-name IsiCifar --checkpoint 4000 --hessian bound_calculation/hessian/hess_isi_cifar_4000.npz --sample-loc bound_calculation/samle_isi_cifar_4000/ ----num-iterations 100 500 1000 4000 --checkpoint-loc bound_calculation/checkpoints_isi_cifar_4000/ --gpus 0 --results-loc bound_calculation/isi_cifar_results/ --log-level INFO
```

We also provide a script in example_bound.sh that uses the STR Mnist model as a toy example.
You can try running this to ensure the repo has been installed correctly.
However, this script can still take several hours to complete, as many of the computations are intensive even for small examples.

NOTE: Currently, the computation requires use of a GPU.
This may be changed in the future.

## In House Models

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
