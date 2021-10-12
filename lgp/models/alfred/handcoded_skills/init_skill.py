from typing import Dict

from lgp.abcd.skill import Skill

from lgp.env.alfred.alfred_action import AlfredAction
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr

from lgp.flags import LONG_INIT

if LONG_INIT:
    INIT_SEQUENCE = ["LookDown"] + ["RotateLeft"] * 4 + ["LookUp"] * 3 + ["RotateLeft"] * 4 + ["LookDown"] * 2 + ["Stop"]
else:
    INIT_SEQUENCE = ["RotateLeft"] * 4 + ["Stop"]


class InitSkill(Skill):
    def __init__(self):
        super().__init__()
        self._reset()
        print(f"Init skill with sequence of length: {INIT_SEQUENCE}")

    @classmethod
    def sequence_length(cls):
        return len(INIT_SEQUENCE) - 1 # -1 because of the Stop action

    def start_new_rollout(self):
        self._reset()

    def _reset(self):
        self.count = 0
        self.trace = {}

    def get_trace(self, device="cpu") -> Dict:
        return self.trace

    def clear_trace(self):
        self.trace = {}

    def has_failed(self) -> bool:
        return False

    def set_goal(self, hl_action : AlfredSubgoal):
        self._reset()

    def act(self, state_repr: AlfredSpatialStateRepr) -> AlfredAction:

        if self.count >= len(INIT_SEQUENCE):
            raise ValueError("Init skill already output a Stop action! No futher calls allowed")
        action = AlfredAction(action_type=INIT_SEQUENCE[self.count], argument_mask=AlfredAction.get_empty_argument_mask())
        self.count += 1
        return action