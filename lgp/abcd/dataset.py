from abc import abstractmethod
from typing import Dict, List

from torch.utils.data.dataset import Dataset


class ExtensibleDataset(Dataset):

    @abstractmethod
    def load(self, directory: str, cap: int = -1):
        ...

    @abstractmethod
    def add_rollout(self, rollout: List[Dict]):
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Name to identify this model as opposed to other models used within the same agent"""
        ...

    @abstractmethod
    def collate_fn(self, list_of_examples: List[Dict])->Dict:
        ...
