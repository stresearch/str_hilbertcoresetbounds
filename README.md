# ML Bound Calculation

There are some scripts in here that allow for calculation of performance bounds on a given network

## Initialization 

Before running the code, clone the git repo.In teh following steps, please replace REPO_LOC with the path to the base of this repo.

Run the following command to set up the repo:

```bash
pip install -r requirements.txt
```
You will also need the TA1 models in order to run the code. 
They are located in the project share folder.
We will add information about how to access them and provide location information for other teams if/when this repo is distributed.

# Hilbert Coresets

These bounds are based on the Hilbert Coreset formulation in [this] pape (link wiill be added in future). We provide teh code to run teh Hilbert coreset calculations used in that paper.

## Usage

You can run any of the provided models to calculate the bounds with Hilbert coresets. 

Example:

```bash
python3 coreset_calc_script.py --model-name IsiCifar --checkpoint 4000 --hessian hess_isi_cifar_4000.npz --sample-loc ../samle_isi_cifar_4000/ ----num-iterations 100 500 1000 4000 --checkpoint-loc ../checkpoints_isi_cifar_4000/ --gpus 0 --results-loc results/ --log-level INFO
```

We also provide a script in example_bound.sh that uses the STR Mnist model as a toy example.
You can try running this to ensure the repo has been installed correctly.
However, this script can still take several hours to complete, as many of the computations are intensive even for small examples.

## In House Models

The Model module provides an interface between the STR bound calculation code and a model. Documentation for the functions is provided in the module. To calculate bounds for your own in-house model, implement a class that extends teh Model interface. There are 4 key functions to implement:

* get_model_weights: Get the parameters from the model
* set_model_weights: Manually change teh parameters of the model to new values
* model_eval: Evaluate the model on provided data with given weights
* grad_model_out_weights: Get the gradient of the model with respect to the parameters, evaluated at given data points

There are example implementations for the provided models.

You must also add your model as an option to run in the model-name command line argument
