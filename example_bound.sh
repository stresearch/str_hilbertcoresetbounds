# A toy script of bound computation
# Uses the STR Mnist model
# Will generate several files in this directory:
# hess.npy: The diagonal of the Hessian
# sample_0.npy: The sample of the loss from the posterior (10, 70000) array
# test_results.npz: The output of the bound computation
# str_lwll.log: The log statements from the computation
echo Even this toy example can easily take over an hour
python3 coreset_calc_script.py -m StrMnist -c 500 -e hess.npy -l . --sigma-prior 0.0001 -j 10 -n 5 -k . -g 0 -r test_results.npz -o Debug
