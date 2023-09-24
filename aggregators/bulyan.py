
import tools
from . import register

import math
import torch

# Optional 'native' module
try:
  import native
except ImportError:
  native = None

# ---------------------------------------------------------------------------- #
# Bulyan GAR class

def aggregate(gradients, f, m=None, **kwargs):
  """ Bulyan over Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    m         Optional number of averaged gradients for Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  n = len(gradients)
  d = gradients[0].shape[0]
  # Defaults
  m_max = n - f - 2
  if m is None:
    m = m_max
  # Compute all pairwise distances
  distances = list([(math.inf, None)] * n for _ in range(n))
  for gid_x, gid_y in tools.pairwise(tuple(range(n))):
    dist = gradients[gid_x].sub(gradients[gid_y]).norm().item()
    if not math.isfinite(dist):
      dist = math.inf
    distances[gid_x][gid_y] = (dist, gid_y)
    distances[gid_y][gid_x] = (dist, gid_x)
  # Compute the scores
  scores = [None] * n
  for gid in range(n):
    dists = distances[gid]
    dists.sort(key=lambda x: x[0])
    dists = dists[:m]
    scores[gid] = (sum(dist for dist, _ in dists), gid)
    distances[gid] = dict(dists)
  # Selection loop
  selected = torch.empty(n - 2 * f - 2, d, dtype=gradients[0].dtype, device=gradients[0].device)
  for i in range(selected.shape[0]):
    # Update 'm'
    m = min(m, m_max - i)
    # print("m = %d" % m)
    # Compute the average of the selected gradients
    scores.sort(key=lambda x: x[0])
    selected[i] = sum(gradients[gid] for _, gid in scores[:m]).div_(m)
    # print("selected[%d] = %r" % (i, selected[i]))
    # Remove the gradient from the distances and scores
    gid_prune = scores[0][1]
    # print("prune %d" % gid_prune)
    scores[0] = (math.inf, None)
    for score, gid in scores[1:]:
      if gid == gid_prune:
        scores[gid] = (score - distance[gid][gid_prune], gid)

  # Coordinate-wise averaged median
  m        = selected.shape[0] - 2 * f
  sorted_m_scores = sorted(scores, key=lambda x: x[0])[:m]
  selected_indecies = [x[1] for x in sorted_m_scores]

  median   = selected.median(dim=0).values
  closests = selected.clone().sub_(median).abs_().topk(m, dim=0, largest=False, sorted=False).indices

  log_file = kwargs.get('log_file', None)
  data_itr = kwargs.get('data_itr', 0)

  if log_file is not None:
      log_file.write("data_itr," + str(data_itr) + ",")
      for i in range(n):
        if i in selected_indecies:
          log_file.write("1,")
        else:
          log_file.write("0,")
      log_file.write("\n")

  closests.mul_(d).add_(torch.arange(0, d, dtype=closests.dtype, device=closests.device))
  avgmed   = selected.take(closests).mean(dim=0)
  # Return resulting gradient

  iter = kwargs.get('iter', 0)
  filepath = kwargs.get('filepath', "/home/data/Garfield/pytorch_impl/applications/Aggregathor/")

  # save the gradient
  # if (iter % 5 == 0):
  #   torch.save(avgmed, filepath + 'Y_bulyan_' + str(iter//5) + '.pt')

  return avgmed

def aggregate_native(gradients, f, m=None, **kwargs):
  """ Bulyan over Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    m         Optional number of averaged gradients for Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  # Defaults
  if m is None:
    m = len(gradients) - f - 2
  # Computation
  return native.bulyan.aggregate(gradients, f, m)

def check(gradients, f, m=None, **kwargs):
  """ Check parameter validity for Bulyan over Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    m         Optional number of averaged gradients for Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return "Expected a list of at least one gradient to aggregate, got %r" % gradients
  if not isinstance(f, int) or f < 1 or len(gradients) < 4 * f + 3:
    return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (f, (len(gradients) - 3) // 4)
  if m is not None and (not isinstance(m, int) or m < 1 or m > len(gradients) - f - 2):
    return "Invalid number of selected gradients, got m = %r, expected 1 ≤ m ≤ %d" % (f, len(gradients) - f - 2)

def upper_bound(n, f, d):
  """ Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this rule.
  Args:
    n Number of workers (Byzantine + non-Byzantine)
    f Expected number of Byzantine workers
    d Dimension of the gradient space
  Returns:
    Theoretical upper-bound
  """
  return 1 / math.sqrt(2 * (n - f + f * (n + f * (n - f - 2) - 2) / (n - 2 * f - 2)))

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "bulyan"
register(method_name, aggregate, check, upper_bound=upper_bound)

# Register aggregation rule (native version, if available)
if native is not None:
  native_name = method_name
  method_name = "native-" + method_name
  if native_name in dir(native):
    register(method_name, aggregate_native, check, upper_bound=upper_bound)
  else:
    tools.warning("GAR %r could not be registered since the associated native module %r is unavailable" % (method_name, native_name))
