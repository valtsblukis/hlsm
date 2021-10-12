from typing import Union
from abc import ABC, abstractmethod
import torch.nn as nn

from lgp.abcd.observation import Observation
from lgp.abcd.repr.state_repr import StateRepr
from lgp.abcd.subgoal import Subgoal


class ObservationFunction(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, observation: Observation, prev_state: Union[StateRepr, None], goal: Union[Subgoal, None]) -> StateRepr:
        ...