from . import register
from . import center_algorithms as ca
from . import center_algorithms_torch as cat
import numpy as np
import torch
from torch import linalg as LA
import torch.nn.functional as F
import sys
# ---------------------------------------------------------------------------- #
# Flag GAR

def aggregate(gradients, **kwargs):
  """ Averaging rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    FlagAggregator gradient
  """
  # print("\nAverage:")
  # print("Length of gradients (Average): " + str(len(gradients)) + "\n")
  # print("Shape of each gradient (Average): " + str(gradients[0].shape) + "\n")

  # avgg = sum(gradients) / len(gradients)
  # normalize gradients to their norm
  # gradients = [g/LA.norm(g) for g in gradients]

  gradients = [F.normalize(g, p=2.0, dim=0) for g in gradients]

  # print("Shape of the Average gradient: " + str(avgg.shape) + "\n")
  # print("Length of Average gradient: " + str(len(avgg)) + "\n")
  
  r = kwargs.get('r_col', 1)
  n_its = 4 #number of iterations for FlagIRLS
  seed = 1 #random seed

  # print("len(gradients): " + str(len(gradients)))
  # print("len(gradients[0]): " + str(len(gradients[0])))
  # print("gradients[0].shape: " + str(gradients[0].shape))

  # mean = torch.mean(torch.stack(gradients), dim=0)
  # print("mean: " + str(mean) + "len(mean)" + str(len(mean)))
  # std = torch.std(torch.stack(gradients), dim=0) + 1e-8
  # print("std: " + str(std) + "len(std)" + str(len(std)))

  # print("Shape of mean: " + str(mean.shape))
  # print("Shape of std: " + str(std.shape))

  # normalize gradients
  # gradients = [torch.transpose((g[None] - mean) / std, 0, 1) for g in gradients]

  # transpose normed using torch
  gradients = [torch.transpose(g[None], 0, 1) for g in gradients]

  # normed = (gradients - mean) / std
  # print("len(normed): "  + str(len(normed)) + " len(normed[0]): " + str(len(normed[0])) + " normed[0].shape" + str(normed[0].shape))

  # print("\nFlagMedian:")
  # print("Length of gradients: " + str(len(gradients)) + "\n")
  # print("Shape of each gradient: "+ str(gradients[0].shape) + "\n")

  flag_median = cat.irls_flag(gradients, r, n_its, 'sine', opt_err = 'sine', seed = seed, **kwargs)[0]
  
  # print("Shape of Flagmedian gradient: "+ str(torch.tensor(flag_median).shape) + "\n")
  # print("Length of Flagmedian gradients: "+ str(len(torch.tensor(flag_median))) + "\n")

  # print shape of flag_median
  # print("Shape of flag_median: " + str(flag_median.shape))

  # flatten flag_median
  # flag_median = torch.flatten(flag_median)
  # print("Shape of flag_median: " + str(flag_median.shape))

  YTX = [flag_median.T@gradients[i] for i in range(len(gradients))]
  YYTX = [flag_median@YTX[i] for i in range(len(YTX))]
  
  # print("Shape of flag_median: " + str(torch.mean(torch.stack(YYTX), dim=0).shape))
  # sys.exit("")
  # return flag_median * std + mean #sum(gradients) / len(gradients)
  
  #return torch.mean(torch.stack(YYTX), dim=0)

  return torch.sum(torch.stack(YYTX), dim=0).div(r)

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
register("flag_median", aggregate, check)
