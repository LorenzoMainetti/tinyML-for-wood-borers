import inspect
import sys
from typing import Callable

import tensorflow as tf


def has_arg(function: Callable,
            argument: str
            ) -> bool:
    """
    Checks whether callable accepts given keyword argument

    Parameters
    ----------
    function: callable
      Function which arguments we are checking

    argument: str
      Name of the argument

    Returns
    -------
    : bool
      True if user provided argument is one of the keyword arguments
      of the given function

    References
    ----------
    https://github.com/keras-team/keras/pull/7035/
    """
    if sys.version_info < (3, 3):
        arg_spec = inspect.getfullargspec(function)
        return argument in arg_spec.args
    else:
        signature = inspect.signature(function)
        return signature.parameters.get(argument) is not None


def regularize_norm_diff(variables, norm):
    """
    Regularize l2/l1 distance between variables

    Parameters
    ----------
    variables: List of lists
       List of variables

    norm: str
       Norm, either L2 or L1

    Returns
    -------
    loss: tf.float32
       l2 or l1 norm of difference normalized by number of parameters
    """
    if norm.lower() == "l2":
        norm_func = tf.square
    elif norm.lower() == "l1":
        norm_func = tf.abs
    else:
        raise ValueError("Norm should be either L1 or L2")

    loss = 0.
    for tensor in zip(*variables):
        for var_one, var_two in zip(tensor, tensor[1:]):
            diff = var_one - var_two
            n_params = tf.cast(tf.size(diff), dtype=tf.float32)
            loss += tf.reduce_sum(norm_func(diff)) / n_params
    return loss
