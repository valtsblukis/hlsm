import torch
from abc import ABC, abstractmethod
import torch.nn as nn

from lgp.abcd.repr.state_repr import StateRepr
from lgp.abcd.repr.task_repr import TaskRepr


class ValueFunction(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, state_repr: StateRepr, task_repr: TaskRepr) -> torch.tensor:
        ...