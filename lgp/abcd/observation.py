import torch
from typing import Iterable
from abc import ABC, abstractmethod


class Observation(ABC):
    def __init__(self):
        ...

    @classmethod
    @abstractmethod
    def collate(cls, observations: Iterable["Observation"]) -> "Observation":
        """
        Creates a single "observation" that represents a batch of observations
        """
        ...

    @abstractmethod
    def represent_as_image(self) -> torch.tensor:
        ...

    @abstractmethod
    def to(self, device) -> "Observation":
        """Moves self to the given Torch device, and returns self"""
        ...