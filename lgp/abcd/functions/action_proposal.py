from abc import ABC, abstractmethod
from typing import Iterable
import torch.nn as nn
import torch

from lgp.abcd.action import Action
from lgp.abcd.repr.state_repr import StateRepr
from lgp.abcd.repr.task_repr import TaskRepr
from lgp.abcd.repr.action_repr import ActionRepr
from lgp.abcd.repr.action_distribution import ActionDistribution


class ActionProposal(nn.Module, ABC):
    class ModelState(ABC):
        @abstractmethod
        def __init__(self):
            ...

        @classmethod
        @abstractmethod
        def blank(cls) -> "ActionProposal.ModelState":
            ...

    """
    Given a current state s_t, proposes an action distribution that makes sense.
    """
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def set_state(self, state : "ActionProposal.ModelState"):
        ...

    @abstractmethod
    def get_state(self):
        ...

    #@abstractmethod
    def _get_state_for_action(self, action: Action) -> "ActionProposal.ModelState":
        """
        Given that "action" was executed in the environment since the last action proposal, compute the next
        hidden state
        """
        ...

    @abstractmethod
    def reset_state(self):
        ...

    @abstractmethod
    def forward(self, state: StateRepr, task: TaskRepr, prev_action: ActionRepr) -> ActionDistribution:
        ...

    def _uniform(self, length : int):
        probs = torch.tensor([1 / float(length) for _ in range(length)])
        return probs

    def mle(self,
            state: StateRepr,
            task: TaskRepr,
            model_state: ModelState):
        action_distr: ActionDistribution = self(state, task, model_state)
        action = action_distr.mle()
        return action

    def propose(self, state: StateRepr,
                task: TaskRepr,
                prev_action : ActionRepr,
                with_replacement : bool,
                max_num_actions : int,
                model_state : "ActionProposal.ModelState"):
        # TODO: IMPORTANT! Use the model_state
        action_distr: ActionDistribution = self(state, task, prev_action)
        action_opts: Iterable[Action] = action_distr.sample(max_num_actions, with_replacement=with_replacement)
        spatial_arg_distrs = action_distr.last_sample_spatial_args

        # Get action proposal states for each action
        proposal_states = [self._get_state_for_action(action) for action in action_opts]

        if with_replacement:
            # If sampling with replacement, more common actions are already oversampled - use uniform probs
            action_probs = self._uniform(max_num_actions)
        else:
            # If sampling without replacement, use
            action_probs = action_distr.probs(list(action_opts))
        return action_opts, action_probs, spatial_arg_distrs, proposal_states