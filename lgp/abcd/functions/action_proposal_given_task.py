from abc import ABC, abstractmethod
import torch.nn as nn

from lgp.abcd.repr.task_repr import TaskRepr
from lgp.abcd.repr.action_distribution import ActionDistribution


class ActionProposalGivenTask(nn.Module, ABC):
    """
    Given a current state s_t, proposes an action distribution that makes sense.
    """
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, task: TaskRepr) -> ActionDistribution:
        ...