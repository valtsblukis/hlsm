import random
from lgp.abcd.task import Task
from lgp.env.blockworld.state.world import World
from lgp.env.blockworld.state.room import Room
import lgp.env.blockworld.config as config


def get_vocabulary_old():
    color_names = config.COLORS
    first_tokens = [
        "<null>"
    ]  # NULL has to go first to get an index zero and later allow easier optimizations
    additional_words = [
        "pick",
        "up",
        "item",
        "room",
        "go",
        "to"
    ]
    special_tokens = [
        "<sos>",
        "<eos>",
        "<unk>"
    ]
    full_vocab = first_tokens + color_names + additional_words + special_tokens
    numbered_vocab = {w: i for i, w in enumerate(full_vocab)}
    return numbered_vocab


def get_vocabulary():
    color_names = config.COLORS
    first_tokens = [
        "<null>"
    ]  # NULL has to go first to get an index zero and later allow easier optimizations
    additional_words = [
        "move"
        "pick",
        "up",
        "item",
        "room",
        "go",
        "to"
    ]
    special_tokens = [
        "<sos>",
        "<eos>",
        "<unk>"
    ]
    full_vocab = first_tokens + color_names + additional_words + special_tokens
    numbered_vocab = {w: i for i, w in enumerate(full_vocab)}
    return numbered_vocab


class BwTask(Task):
    def __init__(self):
        super().__init__()
        ...

    def compute_reward(self, state: World, next_state: World) -> float:
        """
        Given a state transition, award a reward if this transition measurably proceeds towards completing the task.
        :param state:
        :param next_state:
        :return:
        """
        return 0.0

    def check_goal_conditions(self, state: World) -> bool:
        ...


class BwMoveTask(BwTask):
    def __init__(self, item_color: str, room_color: str):
        super().__init__()
        self.item_color = item_color
        self.room_color = room_color

    @classmethod
    def make_random(cls):
        item_color = random.choice(config.COLORS)
        room_color = random.choice(config.COLORS)
        return cls(item_color, room_color)

    def _agent_has_correct_item(self, state: World):
        matching_items = [item for item in state.inventory if item.color == self.item_color]
        return len(matching_items) == 1

    def _target_room_has_target_item(self, state: World):
        for room in state.rooms:
            if room.color == self.room_color:
                if self.item_color in [item.color for item in room.items]:
                    return True
        return False

    def compute_reward(self, state: World, next_state: World) -> float:
        """
        Given a state transition, award a reward if this transition measurably proceeds towards completing the task.
        :param state:
        :param next_state:
        :return:
        """
        STEP_R = -0.05
        already_solved = self.check_goal_conditions(state)
        #now_solved = self.check_goal_conditions(next_state)
        prev_holding_item = self._agent_has_correct_item(state)
        now_holding_item = self._agent_has_correct_item(next_state)

        # Agent already stopped, no further reward needed
        if state.stopped:
            return 0
        # No difference in state means an impossible action was taken
        elif state == next_state:
            return -0.15
        # Goal conditions satisfied
        elif already_solved:
            # Reward for stopping
            if (not state.stopped) and next_state.stopped:
                return 1.0
            # Per-step penalty for doing other things when task is already solved
            else:
                return STEP_R
        # If task not yet solved, reward for picking up the necessary item that has to be moved to the target room
        elif (not prev_holding_item) and now_holding_item:
            return 0.5 + STEP_R
        # If task not yet solved, penalize for dropping the item. This must balance out with the above!
        elif prev_holding_item and not now_holding_item:
            return -0.5 + STEP_R

        # If didn't get any reward, give a negative per-step reward
        else:
            return STEP_R

    def check_goal_conditions(self, state: World) -> bool:
        return self._target_room_has_target_item(state)

    def __str__(self):
        return f"move {self.item_color} item to {self.room_color} room"


class BwPickupTask(BwTask):

    def __init__(self, color: str):
        super().__init__()
        self.color = color

    @classmethod
    def make_random(cls):
        color = random.choice(config.COLORS)
        return cls(color)

    def _room_has_item(self, state: World):
        current_room : Room = state.get_current_room()
        matching_items = [item for item in current_room.items if item.color == self.color]
        return len(matching_items) > 0

    def compute_reward(self, state: World, next_state: World) -> float:
        """
        Given a state transition, award a reward if this transition measurably proceeds towards completing the task.
        :param state:
        :param next_state:
        :return:
        """
        # int(True) is 1
        # Agent already stopped, no further reward needed
        if state.stopped and next_state.stopped:
            return -0.05

        # No difference in state means an impossible action was taken
        if state == next_state:
            return -0.15

        # Task reward for stopping having picked up the correct item
        met_goal_conditions = int(self.check_goal_conditions(next_state))
        # Agent output a STOP action, reward task completion
        if next_state.stopped and not state.stopped:
            reward_task = met_goal_conditions * 1.0
        else:
            reward_task = 0.0

        # Penalize inefficiency
        reward_time = -0.05

        # Don't penalize "moving to room without item" if the item was collected.
        reward = reward_task + reward_time
        return reward

    def check_goal_conditions(self, state: World) -> bool:
        for item in state.inventory:
            if item.color == self.color:
                return True
        return False

    def __str__(self):
        return f"pick up {self.color} item"


#TASK_TYPES = [BwPickupTask, BwMoveTask]
#TASK_TYPES = [BwMoveTask]
TASK_TYPES = [BwPickupTask]

def generate_random_task():
    TaskTypeCls = random.choice(TASK_TYPES)
    task = TaskTypeCls.make_random()
    return task