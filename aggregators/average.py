

from . import register
# import torch
# ---------------------------------------------------------------------------- #
# Average GAR

def aggregate(gradients, **kwargs):
  """ Averaging rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    Average gradient
  """
  average = sum(gradients) / len(gradients)
  return average

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

def influence(honests, attacks, **kwargs):
  """ Compute the ratio of accepted Byzantine gradients.
  Args:
    honests Non-empty list of honest gradients to aggregate
    attacks List of attack gradients to aggregate
    ...     Ignored keyword-arguments
  """
  return len(attacks) / (len(honests) + len(attacks))

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("average", aggregate, check, influence=influence)
