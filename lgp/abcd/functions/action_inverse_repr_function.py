from abc import ABC, abstractmethod

import torch.nn as nn
from lgp.abcd.action import Action
from lgp.abcd.observation import Observation
from lgp.abcd.repr.action_repr import ActionRepr


class ActionInverseReprFunction(nn.Module, ABC):
    """
    Function that builds a task-conditioned state representation
    """
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, action_repr: ActionRepr, observation: Observation) -> Action:
        ...