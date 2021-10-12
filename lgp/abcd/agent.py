from abc import ABC, abstractmethod
from typing import List, Union, Dict

from lgp.abcd.repr.state_repr import StateRepr
from lgp.abcd.task import Task
from lgp.abcd.observation import Observation
from lgp.abcd.action import Action


class Agent(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def get_trace(self, device="cpu") -> Dict:
        return {}

    @abstractmethod
    def clear_trace(self):
        ...

    def action_execution_failed(self):
        # Tell the agent that the most recently predicted action has failed
        ...

    @abstractmethod
    def start_new_rollout(self, task: Task, state_repr: StateRepr = None):
        # Optionally take state_repr argument to allow switching instructions while keeping the map
        ...

    def finalize(self, total_reward: float):
        """A chance for the agent to wrap up after a task is done (e.g. by saving trace data or what not)"""
        ...

    @abstractmethod
    def act(self, observation_or_state_repr: Union[Observation, StateRepr]) -> Action:
        ...

from lgp.abcd.model import LearnableModel

class TrainableAgent(Agent):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_learnable_models(self) -> List[LearnableModel]:
        ...
