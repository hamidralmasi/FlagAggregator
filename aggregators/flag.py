
from . import register
from . import center_algorithms as ca
from . import center_algorithms_torch as cat
import numpy as np
import pandas as pd
import torch
from torch import linalg as LA
import torch.nn.functional as F
import sys
import csv
# ---------------------------------------------------------------------------- #
# Flag GAR

def aggregate(gradients, **kwargs):
  """ Averaging rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    flag_aggregator gradient
  """

  avgg = sum(gradients) / len(gradients)
  gradients = [F.normalize(g, p=2.0, dim=0) for g in gradients]
  r = kwargs.get('r_col', 1)

  n_its = 1 #number of iterations for FlagIRLS
  seed = 1 #random seed

  gradients = [torch.transpose(g[None], 0, 1) for g in gradients]


  flag_median = cat.irls_flag(gradients, r, n_its, 'sine', opt_err = 'sine', seed = seed, **kwargs)[0]

  iter = kwargs.get('iter', 0)
  values = [(gradients[i].T@flag_median@flag_median.T@gradients[i])/(gradients[i].T@gradients[i]) for i in range(len(gradients))]

  # Assuming that 'values' is your array of 15 CUDA tensors

  header = ['w' + str(i+1) for i in range(15)]
  with open('/home/data/Garfield/pytorch_impl/applications/Aggregathor/values_optimal.csv', 'a+', newline='') as f:
    writer = csv.writer(f)
    if iter == 1:
      f.truncate(0)
      writer.writerow(header)
    else:
      values_np = [values[i].cpu().numpy().flatten().tolist() for i in range(len(values))]
      values_np = [str(val).strip('[]') for val in values_np]
      writer.writerow(values_np)

  YTX = [flag_median.T@gradients[i] for i in range(len(gradients))]
  YYTX = [flag_median@YTX[i] for i in range(len(YTX))]

  Y_star = torch.sum(torch.stack(YYTX), dim=0).div(r)
  # save Y_star with lambda, iteration
  # torch.save(Y_star, 'Y_star.pt')

  iter = kwargs.get('iter', 0)
  lambda_ = kwargs.get('lambda_', 0)
  filepath = kwargs.get('filepath', "/home/data/Garfield/pytorch_impl/applications/Aggregathor/")

  # save the gradient
  # if (iter % 5 == 0):
  #   torch.save(Y_star, filepath + 'Y_flag_median_' + str(lambda_) + '_' + str(iter//5) + '.pt')

  return Y_star
  #return torch.mean(torch.stack(YYTX), dim=0)

def check(gradients, **kwargs):
  """ Check parameter validity for the averaging rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return "Expected a list of at least one gradient to aggregate, got %r" % gradients


# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("flag", aggregate, check)
