# Flag Aggregator

This folder contains the implementation of the [Flag Aggregator](https://openreview.net/forum?id=7avlrpzWqo) on PyTorch.


## Requirements
Our code was tested with these settings:
* torch (1.13.1+cu117)
* torchvision (0.14.1+cu117)
* Python (3.8.15)
* Numpy (1.23.5)
* Scipy (1.10.1)

## Installation
The following steps should be applied for **ALL** machines that are going to be involved in aggregating using FA, Bulyan, Multi-Krum, etc.

1. Follow [PyTorch installation guide](https://pytorch.org/) (depending on your environment and preferences).

2. Install the other required packages by running `pip install numpy==1.23.5 scipy==1.10.1` (this command works also for `conda` users).

## Structure

* `aggregators`

   The main components of the library.

   Each file contains one main module for aggregation.

* `Main (current) directory for running the experiments.`

   1. `run_exp.sh`: an example script to automatically deploy the training on multiple nodes.

   2. `kill.sh`: a script to end the deployment and do the cleaning.

   3. `trainer.py`: the implementation of the robust distributed training experiment using the Garfield library in `garfieldpp` folder.

   4. `workers` and `servers`: the list of the workers and parameter server machines IPs that should contribute to training for the corresponding application.

## Notes on the `trainer.py` script
* Single server, multiple workers setup.

* Deployment requires running `trainer.py` on multiple machines.

* Arguments:

- `--master` (str, default=""): Master node in the deployment. This node takes rank 0, usually the first PS.
- `--rank` (int, default=0): Rank of a process in a distributed setup.
- `--dataset` (str, default="cifar10"): Dataset to be used, e.g., mnist, cifar10, tinyimagenet, ...
- `--batch` (int, default=128): Minibatch size to be employed by each worker.
- `--num_ps` (int, default=1): Number of parameter servers in the deployment (Vanilla AggregaThor uses 1 ps).
- `--num_workers` (int, default=1): Number of workers in the deployment.
- `--fw` (int, default=0): Number of declared Byzantine workers.
- `--fps` (int, default=0): Number of declared Byzantine parameter servers (Vanilla AggregaThor does not assume Byzantine servers).
- `--model` (str, default='convnet'): Model to be trained, e.g., convnet, resnet18, resnet50,...
- `--loss` (str, default='nll'): Loss function to optimize against.
- `--optimizer` (str, default='sgd'): Optimizer to use.
- `--opt_args` (dict, default={'lr':'0.1'}): Optimizer arguments; passed in dict format, e.g., '{"lr":"0.1"}'
- `--num_iter` (int, default=5000): Number of training iterations to execute.
- `--gar` (str, default='average'): Aggregation rule for aggregating gradients.
- `--acc_freq` (int, default=100): The frequency of computing accuracy while training.
- `--bench` (bool, default=False): If True, time elapsed in each step is printed.
- `--log` (bool, default=False): If True, accumulated loss at each iteration is printed.
- `--lr_update_freq` (int, default=10): Every what number of epochs should adjust learning rate.
- `--port` (int, default=29500): Port to be used for RPC communication.
- `--attack` (str, default='random'): Attack to be used by Byzantine workers, e.g., random, drop, ...
- `--r_col` (int, default=1): FlagMedian parameter for r-dimensional subspace of R^n, [Y∗] ∈ Gr(r,n), that is in some sense the center of points.
- `--lambda_` (float, default=0): Flag aggregator regularization hyperparameter serving as a coefficient for pairwise terms in the objective function.
- `--augmenteddataset` (str, default='none'): Which augmented dataset to use, cifar10noisy, mnistnoisy tinyimagenetnoisy.
- `--augmentedfolder` (str, default='none'): Which augmented folder to use, lv, half, onethird, train.
- `--seed` (int, default=1001): Seed for reproducibility.
- `--savedir` (str, default=''): Which folder should the results go to.

## Starting/Stopping an experiment

* The script `run_exp.sh` deploys the robust aggregation experiment on multiple machines and it can be used as follows:

  1. Create two files, `servers` and `workers`, and fill them with hostnames or IPs of nodes which should contribute to the experiment. The first file, `servers` should contain only one line, i.e., one host, where the second file, `workers` should contain as many lines as the number of hosts (each line should contain one hostname or IP).

  2. Run `./run_exp.sh`. Note that the parameters to be given to `trainer.py` are hard-coded in this script file. Users should feel free to change them to their favorable choices.

  3. Run `./kill.sh` to clean up.

## Useful practical general notes
1. The repo should be cloned on all nodes contributing to the experiment (NFS would be a good choice for that purpose).

2. The bash scripts (`run_exp.sh` and `kill.sh`) require **password-less** ssh access among machines contributing to the distributed setup.

3. The paths inside `run_exp` scripts and `garfieldpp/dataset.py` should be updated according to your file system and datasets location.

## Citation
```bib
@inproceedings{
almasi2024flag,
title={Flag Aggregator: Scalable Distributed Training under Failures and Augmented Losses using Convex Optimization},
author={Hamidreza Almasi and Harsh Mishra and Balajee Vamanan and Sathya N. Ravi},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=7avlrpzWqo}
}
```

Correspondence to: Hamidreza Almasi <halmas3@uic.edu>.
