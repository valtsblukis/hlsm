from abc import ABC, abstractmethod

import torch.nn as nn
from lgp.abcd.action import Action
from lgp.abcd.observation import Observation
from lgp.abcd.repr.action_repr import ActionRepr


class ActionReprFunction(nn.Module, ABC):
    """
    Function that builds an action representation conditioned on the corresponding observation
    """
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, action: Action, observation: Observation) -> ActionRepr:
        ...