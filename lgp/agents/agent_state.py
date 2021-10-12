from typing import Union, List

from lgp.abcd.action import Action
from lgp.abcd.repr.action_repr import ActionRepr
from lgp.abcd.repr.state_repr import StateRepr
from lgp.abcd.repr.task_repr import TaskRepr


class AgentState:
    """
    The full "state" of the agent. When starting a new task / rollout, clearing this state
    is sufficient to completely reset the agent and make sure there is no residual state.
    # TODO: Make sure that's actually the case. Currently it's not!!
    """
    def __init__(self, task: TaskRepr):
        self.task_repr : TaskRepr = task
        self.prev_state : Union[StateRepr, None] = None
        self.action_history : List[Action] = []
        self.timestep = 0
        self.task = None

    def add_action(self, action: Action):
        self.action_history.append(action)
