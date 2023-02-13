# Flag Aggregator

The implementation of the Flag Aggregator on PyTorch.


## Requirements
Our code was tested with these settings:
* torch (1.8.1)
* torchvision (0.9.1)
* Python (3.6.10)
* Numpy (1.19.5)
* Scipy (1.5.2)

## Installation
The following steps should be applied for **ALL** machines that are going to be involved in aggregating using FA, Bulyan, or MultiKrum.

1. Follow [PyTorch installation guide](https://pytorch.org/) (depending on your environment and preferences).

2. Install the other required packages by running `pip install numpy==1.19.5 scipy==1.5.2` (this command works also for `conda` users).

## Structure

* `aggregators`

   The main components of the library.

   Each file contains one main module for aggregation.

* `Main (current) directory for running the experiments.`

   1. _Symlinks_ to the used modules from the `libs` directory.

   2. `run_exp.sh`: an example script to automatically deploy the training on multiple nodes.

   3. `kill.sh`: a script to end the deployment and do the cleaning.

   4. `trainer.py`: the implementation of the robust distributed training experiment using the Garfield library.

   5. `workers` and `servers`: the list of the workers and parameter server machines that should contribute to training for the corresponding application.

## Notes on the `trainer.py` script
* Single server, multiuple workers setup.

* Deployment requires running `trainer.py` on multiple machines.

```
usage: trainer.py [-h] [--master MASTER] [--rank RANK] [--dataset DATASET]
                  [--batch BATCH] [--num_workers NUM_WORKERS]
                  [--fw FW] [--model MODEL]
                  [--loss LOSS] [--optimizer OPTIMIZER]
                  [--opt_args OPT_ARGS] [--num_iter NUM_ITER] [--gar GAR]
                  [--acc_freq ACC_FREQ] [--bench BENCH] [--log LOG]

optional arguments:
  -h, --help            Show this help message and exit.
  --master MASTER       Master node in the deployment. This node takes rank 0, usually the PS.
  --rank RANK           Rank of a process in the distributed setup.
  --dataset DATASET     Dataset to be used, e.g., mnist, cifar10.
  --batch BATCH         Minibatch size to be employed by each worker.
  --num_workers NUM_WORKERS
                        Number of workers in the deployment.
  --fw FW               Number of declared Byzantine workers.
  --model MODEL         Model to be trained, e.g., convnet, cifarnet, resnet.
  --loss LOSS           Loss function to optimize.
  --optimizer OPTIMIZER
                        Optimizer to use.
  --opt_args OPT_ARGS   Optimizer arguments; passed in dict format, e.g., '{"lr":"0.1"}'
  --num_iter NUM_ITER   Number of training iterations to execute.
  --gar GAR             Aggregation rule for aggregating gradients.
  --acc_freq ACC_FREQ   The frequency of computing accuracy while training.
  --bench BENCH         If True, time elapsed in each step is printed.
  --log LOG             If True, accumulated loss at each iteration is printed.

```

* The script `run_exp.sh` deploys the robust aggregation experiment on multiple machines and it can be used as follows:

  1. Create two files, `servers` and `workers`, and fill them with hostnames of nodes which should contribute to the experiment. The first file, `servers` should contain only one line, i.e., one host, where the second file, `workers` should contain as many lines as the number of hosts (each line should contain one hostname).

  2. Run `./run_exp.sh`. Note that the parameters to be given to `trainer.py` are hard-coded in this script file. Users should feel free to change them to their favorable choices.

  3. Run `./kill.sh` to clean up.

## Useful practical general notes
1. The repo should be cloned on all nodes contributing to the experiment (NFS would be a good choice for that purpose).

2. The bash scripts (`run_exp.sh` and `kill.sh`) require **password-less** ssh access among machines contributing to the distributed setup.


## Citation
Almasi H, Mishra H, Vamanan B, Ravi SN. Flag Aggregator: Scalable Distributed Training under Failures and Augmented Losses using Convex Optimization. arXiv preprint arXiv:xxxx.yyyyy. 2023 Feb 12.

Correspondence to: Hamidreza Almasi <halmas3@uic.edu>.
