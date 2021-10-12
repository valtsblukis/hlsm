from typing import Dict, List, Type, Union

from lgp.abcd.task import Task
from lgp.abcd.agent import Agent
from lgp.abcd.subgoal import Subgoal
from lgp.abcd.action import Action
from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction
from lgp.abcd.repr.state_repr import StateRepr
from lgp.abcd.skill import Skill

from lgp.env.alfred.alfred_action import AlfredAction


class HierarchicalAgent(Agent):
    def __init__(self,
                 highlevel_agent : Agent,
                 skillset : Dict[str, Skill],
                 observation_function: ObservationFunction,
                 action_class : Type[Action]):
        super().__init__()
        self.ActionCls = action_class
        self.skillset = skillset
        self.hl_agent = highlevel_agent
        self.observation_function = observation_function

        # State:
        self.current_skill : Union[Skill, None] = None
        self.current_goal : Union[Subgoal, None] = None
        self.state_repr = None
        self.initialized = False
        self.count = 0

    def start_new_rollout(self, task: Task, state_repr: StateRepr = None):
        self.hl_agent.start_new_rollout(task, state_repr)
        for skill_name, skill in self.skillset.items():
            skill.start_new_rollout()
        self.state_repr = state_repr
        self.initialized = False
        self.current_skill = None
        self.current_goal = None
        self.count = 0

    def finalize(self, total_reward):
        self.hl_agent.finalize(total_reward)

    def get_trace(self, device="cpu"):
        skill_traces = {}
        for skillname, skill in self.skillset.items():
            skill_traces[skillname] = skill.get_trace(device)
        trace = {
            "hl_agent": self.hl_agent.get_trace(device),
            "obs_func": self.observation_function.get_trace(device),
            "skills": skill_traces
        }
        return trace

    def clear_trace(self):
        for skill in self.skillset.values():
            skill.clear_trace()
        self.hl_agent.clear_trace()
        self.observation_function.clear_trace()

    def act(self, observation: Observation) -> Action:
        self.state_repr = self.observation_function(observation, self.state_repr, goal=self.current_goal)

        if not self.initialized:
            action = self.skillset["init"].act(self.state_repr)
            if action.is_stop():
                self.initialized = True
            else:
                return action

        # Keep
        while True:
            if self.current_skill is None:
                hl_action: Subgoal = self.hl_agent.act(self.state_repr)
                print(f"HierarchicalAgent: {str(hl_action)}")
                # If the high-level policy signals a stop, we emit a stop action
                if hl_action.is_stop():
                    return self.ActionCls.stop_action()
                self.current_skill = self.skillset[hl_action.type_str()]
                self.current_skill.set_goal(hl_action)
                self.current_goal = hl_action
            ll_action: Action = self.current_skill.act(self.state_repr)
            if self.current_skill.has_failed():
                self.hl_agent.action_execution_failed()

            # If a low-level policy signals a stop, it indicates that the skill has completed it's job
            # invoke the high-level policy to decide the next action
            if ll_action.is_stop():
                self.current_skill = None
            else:
                break
        return ll_action