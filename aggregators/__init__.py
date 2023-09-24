import pathlib
import torch

import tools

# ---------------------------------------------------------------------------- #
# Automated GAR loader

def make_gar(unchecked, check, upper_bound=None, influence=None):
  """ GAR wrapper helper.
  Args:
    unchecked   Associated function (see module description)
    check       Parameter validity check function
    upper_bound Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this aggregation rule: (n, f, d) -> float
    influence   Attack acceptation ratio function
  Returns:
    Wrapped GAR
  """
  # Closure wrapping the call with checks
  def checked(**kwargs):
    # Check parameter validity
    message = check(**kwargs)
    if message is not None:
      raise tools.UserException("Aggregation rule %r cannot be used with the given parameters: %s" % (name, message))
    # Aggregation (hard to assert return value, duck-typing is allowed...)
    return unchecked(**kwargs)
  # Select which function to call by default
  func = checked if __debug__ else unchecked
  # Bind all the (sub) functions to the selected function
  setattr(func, "check", check)
  setattr(func, "checked", checked)
  setattr(func, "unchecked", unchecked)
  setattr(func, "upper_bound", upper_bound)
  setattr(func, "influence", influence)
  # Return the selected function with the associated name
  return func

def register(name, unchecked, check, upper_bound=None, influence=None):
  """ Simple registration-wrapper helper.
  Args:
    name        GAR name
    unchecked   Associated function (see module description)
    check       Parameter validity check function
    upper_bound Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this aggregation rule: (n, f, d) -> float
    influence   Attack acceptation ratio function
  """
  global gars
  # Check if name already in use
  if name in gars:
    tools.warning("Unable to register %r GAR: name already in use" % name)
    return
  # Export the selected function with the associated name
  gars[name] = make_gar(unchecked, check, upper_bound=upper_bound, influence=influence)

# Registered rules (mapping name -> aggregation rule)
gars = dict()

# Load all local modules
with tools.Context("aggregators", None):
  tools.import_directory(pathlib.Path(__file__).parent, globals())

# Bind/overwrite the GAR name with the associated rules in globals()
for name, rule in gars.items():
  globals()[name] = rule
