from typing import Iterable, List
from abc import ABC, abstractmethod

import torch
from lgp.abcd.action import Action


class ActionDistribution(ABC):

    def __init__(self):
        ...

    @classmethod
    def from_action(cls, action: Action) -> "ActionDistribution":
        ...

    @classmethod
    @abstractmethod
    def collate(cls, actions: Iterable["Action"]):
        """
        Creates a single ActionDistribution that represents a batch of actions
        """
        ...

    @abstractmethod
    def mle(self) -> Action:
        ...

    @abstractmethod
    def sample(self, n=1, with_replacement=True) -> List[Action]:
        """
        Samples an action from this distribution
        :return:
        """
        ...

    @abstractmethod
    def probs(self, actions: List[Action]) -> torch.tensor:
        """
        Returns a tensor of probabilities corresponding to these actions under this distribution
        :param actions:
        :return:
        """
        ...

    @abstractmethod
    def represent_as_image(self) -> torch.tensor:
        ...