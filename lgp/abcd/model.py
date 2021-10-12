from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class LearnableModel(nn.Module, ABC):
    """
    Represents a model that can be trained on batches of transitions.
    """

    @abstractmethod
    def loss(self, batch: Dict) -> (torch.tensor, Dict):
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Name to identify this model as opposed to other models used within the same agent"""
        ...

