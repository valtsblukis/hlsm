from typing import Dict
import torch
import random
import itertools

from lgp.abcd.agent import Agent
from lgp.abcd.repr.state_repr import StateRepr

from lgp.env.alfred.alfred_observation import AlfredObservation
from lgp.env.alfred.tasks import AlfredTask
from lgp.env.alfred.alfred_action import AlfredAction, ACTION_TYPES

from lgp.models.alfred.handcoded_skills.init_skill import InitSkill

import lgp.env.blockworld.config as config


class DemonstrationReplayAgent(Agent):
    def __init__(self):
        super().__init__()
        self.actions = None
        self.init_skill = InitSkill()
        self.initialized = False
        self.current_step = 0

    def get_trace(self, device="cpu") -> Dict:
        return {}

    def clear_trace(self):
        ...

    def start_new_rollout(self, task: AlfredTask, state_repr: StateRepr = None):
        api_ish_actions = task.traj_data.get_api_action_sequence()
        self.actions = [AlfredAction(a["action"], torch.from_numpy(a["mask"]) if a["mask"] is not None else None) for a in api_ish_actions]
        self.current_step = 0
        self.initialized = False
        self.init_skill.start_new_rollout()

    def act(self, observation: AlfredObservation) -> AlfredAction:
        # First run the init skill until it stops
        if not self.initialized:
            action = self.init_skill.act(...)
            if action.is_stop():
                self.initialized = True
            else:
                return action

        # Then execute the prerecorded action sequence
        if self.current_step < len(self.actions):
            action = self.actions[self.current_step]
        else:
            action = AlfredAction("Stop", None)

        self.current_step += 1
        return action
