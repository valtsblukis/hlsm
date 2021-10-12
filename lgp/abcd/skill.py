from abc import ABC, abstractmethod
from typing import Dict

from lgp.abcd.action import Action
from lgp.abcd.repr.state_repr import StateRepr


class Skill(ABC):
    """
    Skills differ from Agents in that
    """
    def __init__(self):
        super().__init__()

    def start_new_rollout(self):
        ...

    @abstractmethod
    def get_trace(self, device="cpu") -> Dict:
        # Return a dictionary of outputs (e.g. tensors, arrays) that illustrate internal reasoning of the skill
        ...

    def clear_trace(self):
        # Clear any traces collected in this rollout to have a clean slate for next rollout or sample
        ...

    @abstractmethod
    def set_goal(self, goal):
        ...

    @abstractmethod
    def act(self, state_repr: StateRepr) -> Action:
        ...

    @abstractmethod
    def has_failed(self) -> bool:
        ...