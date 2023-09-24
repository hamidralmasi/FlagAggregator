# coding: utf-8


#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from time import time
import argparse
import sys
import json
import threading
import csv
import os.path
# from Garfield.pytorch_impl.applications.Aggregathor.garfieldpp.datasets import TINYIMAGENET

import garfieldpp
from garfieldpp.worker import Worker
from garfieldpp.byzWorker import ByzWorker
from garfieldpp.server import Server
from garfieldpp.tools import get_bytes_com,convert_to_gbit, adjust_learning_rate
from datetime import datetime
import aggregators
import numpy as np

CIFAR_NUM_SAMPLES = 50000
TINYIMAGENET_NUM_SAMPLES = 100000
MNIST_NUM_SAMPLES = 60000
#First, parse the inputs
parser = argparse.ArgumentParser(description="AggregaThor implementation using Garfield++ library", formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument("--master",
    type=str,
    default="",
    help="Master node in the deployment. This node takes rank 0, usually the first PS.")
parser.add_argument("--rank",
    type=int,
    default=0,
    help="Rank of a process in a distributed setup.")
parser.add_argument("--dataset",
    type=str,
    default="cifar10",
    help="Dataset to be used, e.g., mnist, cifar10, tinyimagenet, ...")
parser.add_argument("--batch",
    type=int,
    default=128,
    help="Minibatch size to be employed by each worker.")
parser.add_argument("--num_ps",
    type=int,
    default=1,
    help="Number of parameter servers in the deployment (Vanilla AggregaThor uses 1 ps).")
parser.add_argument("--num_workers",
    type=int,
    default=1,
    help="Number of workers in the deployment.")
parser.add_argument("--fw",
    type=int,
    default=0,
    help="Number of declared Byzantine workers.")
parser.add_argument("--fps",
    type=int,
    default=0,
    help="Number of declared Byzantine parameter servers (Vanilla AggregaThor does not assume Byzantine servers).")
parser.add_argument("--model",
    type=str,
    default='convnet',
    help="Model to be trained, e.g., convnet, resnet18, resnet50,...")
parser.add_argument("--loss",
    type=str,
    default='nll',
    help="Loss function to optimize against.")
parser.add_argument("--optimizer",
    type=str,
    default='sgd',
    help="Optimizer to use.")
parser.add_argument("--opt_args",
    type=json.loads,
    default={'lr':'0.1'},
    help="Optimizer arguments; passed in dict format, e.g., '{\"lr\":\"0.1\"}'")
parser.add_argument("--num_iter",
    type=int,
    default=5000,
    help="Number of training iterations to execute.")
parser.add_argument("--gar",
    type=str,
    default='average',
    help="Aggregation rule for aggregating gradients.")
parser.add_argument('--acc_freq',
    type=int,
    default=100,
    help="The frequency of computing accuracy while training.")
parser.add_argument('--bench',
    type=bool,
    default=False,
    help="If True, time elapsed in each step is printed.")
parser.add_argument('--log',
    type=bool,
    default=False,
    help="If True, accumulated loss at each iteration is printed.")
parser.add_argument('--lr_update_freq',
    type=int,
    default=10,
    help="Every what number of epochs should adjust learning rate.")
parser.add_argument('--port',
    type=int,
    default=29500,
    help="Port to be used for RPC communication.")
parser.add_argument('--attack',
    type=str,
    default='random',
    help="Attack to be used by Byzantine workers, e.g., random, drop, ...")
parser.add_argument('--r_col',
    type=int,
    default=1,
    help="FlagMedian parameter for r-dimensional subspace of R^n, [Y∗] ∈ Gr(r,n), that is in some sense the center of points")
parser.add_argument('--lambda_',
    type=float,
    default=0,
    help="Flag aggergator regularization hyperparameter serving as a coefficient for pairwise terms in the objective function")
parser.add_argument('--augmenteddataset',
    type=str,
    default='none',
    help="Which augmented dataset to use, cifar10noisy, mnistnoisy tinyimagenetnoisy")
parser.add_argument('--augmentedfolder',
    type=str,
    default='none',
    help="Which augmented folder to use, lv, half, onethird, train")
parser.add_argument("--seed",
    type=int,
    default=1001,
    help="seed for reproducibility.")
parser.add_argument('--savedir',
    type=str,
    default='',
    help="Which folder should the results go to.")

FLAGS = parser.parse_args(sys.argv[1:])

master = FLAGS.master
assert len(master) > 0

rank = FLAGS.rank
assert rank >= 0

num_ps = FLAGS.num_ps
assert num_ps >= 1
num_workers = FLAGS.num_workers
assert num_workers >= 1
world_size = num_workers + num_ps

fw = FLAGS.fw
assert fw*2 < num_workers
fps = FLAGS.fps
assert fps*2 < num_ps

dataset = FLAGS.dataset
assert len(dataset) > 0
batch = FLAGS.batch
assert batch >= 1
model = FLAGS.model
assert len(model) > 0
loss = FLAGS.loss
assert len(loss) > 0
optimizer = FLAGS.optimizer
assert len(optimizer) > 0
opt_args = FLAGS.opt_args
for k in opt_args:
    opt_args[k] = float(opt_args[k])
assert opt_args['lr']

num_iter = FLAGS.num_iter
assert num_iter > 0

gar = FLAGS.gar
assert len(gar) > 0

acc_freq = FLAGS.acc_freq
assert(acc_freq > 10)
bench = FLAGS.bench
if bench:
  from timeit import timeit
else:
  timeit = None
log = FLAGS.log

port = FLAGS.port
attack = FLAGS.attack

r_col = FLAGS.r_col
assert r_col > 0

lambda_ = FLAGS.lambda_
assert lambda_ >= 0

seed = FLAGS.seed
assert seed > 0

savedir = FLAGS.savedir

augmenteddataset = FLAGS.augmenteddataset
augmentedfolder = FLAGS.augmentedfolder
lr_update_freq = FLAGS.lr_update_freq

mean_loss = 0

#os.environ['CUDA_VISIBLE_DEVICES'] = str((rank%2))
num_samples = CIFAR_NUM_SAMPLES
if dataset == 'cifar10' or dataset == 'cifar10noisy':
    num_samples = CIFAR_NUM_SAMPLES
elif dataset == 'tinyimagenet':
    num_samples = TINYIMAGENET_NUM_SAMPLES
elif dataset == 'mnist' or dataset == 'mnistnoisy' or dataset =='fmnist' or dataset == 'fmnistnoisy':
    num_samples = MNIST_NUM_SAMPLES

print("**** SETUP AT NODE {} ***".format(rank))
print("Number of workers: ", num_workers)
print("Number of servers: ", num_ps)
print("Number of declared Byzantine workers: ", fw)
print("Number of declared Byzantine parameter servers: ", fps)
print("GAR: ", gar)
print("Dataset: ", dataset)
print("Model: ", model)
print("Batch size: ", batch)
print("Loss function: ", loss)
print("Optimizer: ", optimizer)
print("Optimizer Args", opt_args)
print("Benchmarking? ", bench)
print("Logging loss at each iteration?", log)
print("port: ", port)
print("attack: ", attack)
print("r_col: ", r_col)
print("lambda: ", lambda_)
print("seed:", seed)
print("augmenteddataset", augmenteddataset)
print("augmenteddfolder", augmentedfolder)

print("------------------------------------")
sys.stdout.flush()
filepath = "/home/data/Garfield/pytorch_impl/applications/Aggregathor/"
now = datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M-%S")
logfilename = gar + "_" + "weights" + "_n_" + str(num_workers) + "_f_" + str(fw) + "_r_" + str(r_col) + "_lambda_" + str(lambda_) + "_" + str(attack) + "_" + now + ".txt"
checkpointfilename_base = gar + "_model_" + "_dataset_" + str(dataset) + "_batch_" + str(batch) + "_n_" + str(num_workers) + "_f_" + str(fw) + "_r_" + str(r_col) + "_lambda_" + str(lambda_) + "_" + str(attack) + "_epoch_"
lr = opt_args['lr']

output_file = "/home/data/Garfield/pytorch_impl/applications/Aggregathor/NIPS/"+savedir+gar+"_"+str(num_workers)+"_"+str(fw)+"_"+attack+"_"+model+"_"+dataset+"_"+augmenteddataset+"_"+augmentedfolder+"_"+str(batch)+"_"+str(lr)+"_"+str(r_col)+"_"+str(lambda_)+"_"+str(seed)+".csv"
with open(output_file, mode='w') as results_file:
  writer = csv.writer(results_file)
  writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Time'])

#initiating the GAR
gar = aggregators.gars.get(gar)
assert gar is not None



os.environ['MASTER_ADDR'] = master
os.environ['MASTER_PORT'] = str(port)
os.environ['GLOO_SOCKET_IFNAME'] = 'ens3'
os.environ['TP_SOCKET_IFNAME'] = 'ens3'

torch.manual_seed(seed)					#For reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)    #For reproducibility
if bench:
  torch.backends.cudnn.benchmark=True

#convention: low ranks are reserved for parameter servers
if rank < num_ps:
  rpc.init_rpc('ps:{}'.format(rank), rank=rank, world_size=world_size, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method='env://', _transports=["uv"],)) #  rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=80,rpc_timeout=20)
  #Initialize a parameter server and write the training loop
  ps = Server(rank, world_size, num_workers,1, fw, fps,  'worker:', 'ps:', batch, model, dataset, augmentedfolder, optimizer, **opt_args)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(ps.optimizer, milestones=[150, 250, 350], gamma=0.1)		#This line shows sophisticated stuff that can be done out of the Garfield++ library
  start_time = time()
  iter_per_epoch = num_samples//(num_workers * batch)		#this value records how many iteration per sample
  print("One EPOCH consists of {} iterations".format(iter_per_epoch))
  sys.stdout.flush()

  # ps.model.load_state_dict(torch.load("/home/data/Garfield/pytorch_impl/applications/Aggregathor/flag_median_model__dataset_cifar10_batch_200_n_15_f_3_r_10_lambda_0.0_random_epoch_40.pt"))
  # ps.model.eval()

  for i in range(num_iter):
    if i%(iter_per_epoch*lr_update_freq) == 0 and i!=0:			#One hack for better convergence with Cifar10
      lr*=0.2
      adjust_learning_rate(ps.optimizer, lr)
    #training loop goes here
    def train_step():
      global mean_loss
      if bench:
        bytes_rec = get_bytes_com()			#record number of bytes sent before the training step to work as a checkpoint
      with torch.autograd.profiler.profile(enabled=bench) as prof:
        gradients, losses = ps.get_gradients(i, num_workers)     #get_gradients(iter_num, num_wait_wrk)
        mean_loss = np.mean(losses)
        aggr_grad = gar(gradients=gradients, f=fw, log_file=None, data_itr=i, r_col=r_col, lambda_=lambda_, filepath=filepath, iter=i)			#aggr_grad = gar.aggregate(gradients)
        ps.update_model(aggr_grad)
        # if i % (5 * iter_per_epoch) == 0:
        #   torch.save(ps.model.state_dict(), filepath + checkpointfilename_base + str(int(i/iter_per_epoch)) + ".pt")
        if bench:
          print(prof.key_averages().table(sort_by="self_cpu_time_total"))
          bytes_train = get_bytes_com()
          print("Consumed bandwidth in this iteration: {} Gbits".format(convert_to_gbit(bytes_train-bytes_rec)))
#          print("Memory allocated to GPU {} Memory cached on GPU {}".format(torch.cuda.memory_allocated(0), torch.cuda.memory_cached(0)))
          sys.stdout.flush()
    if timeit is not None:
      res = timeit(train_step,number=1)
      print("Training step {} takes {} seconds".format(i,res))
      sys.stdout.flush()
    else:
      train_step()

    if i%iter_per_epoch == 0:
      def test_step():
        acc = ps.compute_accuracy()

        num_epochs = int(i/iter_per_epoch)

        with open(output_file, mode='a') as results_file:
          writer = csv.writer(results_file)
          writer.writerow([num_epochs, "{:.4f}".format(mean_loss), acc, "{:.3f}".format(time()-start_time)])

        sys.stdout.flush()
      if timeit is not None:
        res = timeit(test_step,number=1)
        print("Test step takes {} seconds".format(res))
      else:
        threading.Thread(target=test_step).start()
else:
  rpc.init_rpc('worker:{}'.format(rank-num_ps), rank=rank, world_size=world_size, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method='env://', _transports=["uv"],)) #
  #initialize a worker here
  if (rank > fw):
    Worker(rank, world_size, num_workers, batch, model, dataset, augmentedfolder, loss)
  else:
    if(augmenteddataset != "none"):
      Worker(rank, world_size, num_workers, batch, model, augmenteddataset, augmentedfolder, loss)
    else:
      ByzWorker(rank, world_size, num_workers, batch, model, dataset, augmentedfolder, loss, attack)

rpc.shutdown()
