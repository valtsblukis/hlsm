import random
from typing import Type, Union, List

from lgp.abcd.agent import Agent
from lgp.abcd.action import Action
from lgp.abcd.observation import Observation
from lgp.abcd.task import Task

from lgp.abcd.repr.state_repr import StateRepr
from lgp.abcd.repr.task_repr import TaskRepr
from lgp.abcd.repr.action_repr import ActionRepr

from lgp.abcd.functions.action_proposal import ActionProposal
from lgp.abcd.functions.observation_function import ObservationFunction

from lgp.agents.agent_state import AgentState


class ActionProposalAgent(Agent):

    def __init__(self,
                 action_proposal_model: ActionProposal,
                 observation_function: ObservationFunction,
                 task_repr_cls: Type[TaskRepr],
                 device: str):
        """
        :param agents: A list of agents to create a mixture policy
        :param agent_probs: For
        """
        super().__init__()
        self.action_proposal = action_proposal_model
        self.obs_func = observation_function
        self.TaskReprCls = task_repr_cls

        self.agent_state : Union[AgentState, None] = None
        self.device = device
        self.trace = {}

    def start_new_rollout(self, task: Task, state_repr: StateRepr = None):
        task_repr = self.TaskReprCls.from_task([task], device=self.device)
        self.agent_state = AgentState(task_repr)
        self.action_proposal.reset_state()

    def get_trace(self, device="cpu"):
        self.trace["action_proposal"] = self.action_proposal.get_trace(device)
        return self.trace

    def clear_trace(self):
        self.action_proposal.clear_trace()
        self.trace = {
            "hl_action": None
        }

    def action_execution_failed(self):
        self.action_proposal.action_execution_failed()

    def _select_action(self, state_repr : StateRepr):
        # Find the previous action representation
        #action_history : List[Action] = self.agent_state.action_history

        MLE = True
        if MLE:
            action = self.action_proposal.mle(
                state_repr,
                self.agent_state.task_repr,
                model_state=self.action_proposal.get_state()
            )
            self.trace["hl_action"] = action
            #self.agent_state.add_action(action)
        else:
            # Call the action proposal model
            action_opts, action_probs, action_arg_distrs, model_states = self.action_proposal.propose(
                state_repr,
                self.agent_state.task_repr,
                prev_action=prev_action_repr,
                with_replacement=False,
                max_num_actions=1,
                model_state=self.action_proposal.get_state()
            )
            action = action_opts[0]
        self.action_proposal.log_action(action)
        return action
        #return action_opts, action_probs, action_arg_distrs, model_states

    def act(self, observation_or_state: Union[Observation, StateRepr]) -> Action:
        if isinstance(observation_or_state, Observation):
            observation = observation_or_state
            observation = observation.to(self.device)
            s_0 = self.obs_func(observation, self.agent_state.prev_state)
        else:
            s_0 = observation_or_state
            observation = observation_or_state.observation

        action = self._select_action(s_0)
        self.agent_state.prev_state = s_0
        return action
