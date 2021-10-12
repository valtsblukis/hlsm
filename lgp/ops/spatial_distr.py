"""
PyTorch functions for more easily working with spatial probability distributions.

Most PyTorch probability-related functions assume that a single axis is the domain of the distribution,
treating all other axes as "batch" dimensions. When working with spatial distributions (e.g. over 2D or 3D locations),
the PyTorch built in ops require frequently reshaping the data to flatten/unflatten the spatial dimensions.

Below are common operators on probability distributions that take a dims argument to specifying which tensor axes
contain the domain of the distribution, and will re-shape the data before performing the PyTorch operation.
"""

from typing import Tuple
import torch
import torch.nn.functional as F


# Private implementation:

def _check_dims_consecutive(dims):
    for dim1, dim2 in zip(dims[1:], dims[:-1]):
        assert dim1 - dim2 == 1, "All dimensions to softmax over must be consecutive! Please transpose first."

def _multidim_sm(x: torch.tensor, dims: Tuple[int, ...], log: bool):
    _check_dims_consecutive(dims)
    init_shape = x.shape
    dims = list(sorted(dims))
    new_shape = [d for i, d in enumerate(init_shape) if i not in dims]
    new_shape = new_shape[:dims[0]] + [-1] + new_shape[dims[0]:]
    x = x.reshape(new_shape)
    dim = dims[0]

    if log:
        x = F.log_softmax(x, dim=dim)
    else:
        x = F.softmax(x, dim=dim)

    x = x.reshape(init_shape)
    return x

def _multidim_ce(input: torch.tensor, target: torch.tensor, dims: Tuple[int, ...], input_log: bool = False):
    _check_dims_consecutive(dims)
    x = -target * (input if input_log else torch.log(input))
    # Sum accross probability distribution domain axes
    for _ in range(len(dims)):
        x = x.sum(dims[0])
    return x


# Public ops:

def multidim_softmax(x: torch.tensor, dims: Tuple[int, ...]) -> torch.tensor:
    return _multidim_sm(x, dims, log=False)


def multidim_logsoftmax(x: torch.tensor, dims: Tuple[int, ...]) -> torch.tensor:
    return _multidim_sm(x, dims, log=True)


def multidim_cross_entropy_with_logits(input: torch.tensor, target: torch.tensor, dims: Tuple[int, ...]) -> torch.tensor:
    raise NotImplementedError()
    return _multidim_ce(input, target, dims, log=True)


def multidim_cross_entropy(input: torch.tensor, target: torch.tensor, dims: Tuple[int, ...], input_log) -> torch.tensor:
    return _multidim_ce(input, target, dims, input_log=input_log)