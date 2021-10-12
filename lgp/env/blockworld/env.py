import copy
from typing import Tuple, Dict

from lgp.abcd.env import Env

from lgp.env.blockworld import config as config
from lgp.env.blockworld.bwobservation import BwObservation
from lgp.env.blockworld.state.world import World
from lgp.env.blockworld.bwaction import Action, ActionType, PickupArgument, NavigateArgument
from lgp.env.blockworld.state.direction import Direction
from lgp.env.blockworld.tasks.tasks import BwTask, generate_random_task

from lgp.agents.blockworld.oracle_agent import OracleAgent



class BlockworldEnv(Env):

    def __init__(self, device=None):
        super().__init__()
        self.world = None
        self.task = None
        self.steps = 0
        self.device = device
        self.horizon = config.DEFAULT_HORIZON

    def _transition_function(self, state: World, action: Action) -> (World, float):
        next_state = copy.deepcopy(state)

        # Stop action does not modify the state
        if action.type == ActionType.STOP:
            next_state.stopped = True
            return next_state

        # Move the agent to the neighboring room
        if action.type == ActionType.NAV:
            assert isinstance(action.argument, NavigateArgument), "Incompatible action and argument type"
            room = next_state.get_current_room()
            displacements = {
                Direction.UP: (-1, 0),
                Direction.DOWN: (1, 0),
                Direction.LEFT: (0, -1),
                Direction.RIGHT: (0, 1)
            }
            if action.argument.direction in room.doors:
                new_room_coord = tuple([a + b for a, b in zip(room.coord, displacements[action.argument.direction])])
            else:
                #print(f"Env: Attempted to navigate {action.argument.direction}, but door is not available!")
                new_room_coord = room.coord
            next_state.move_agent_to_room(new_room_coord)

        # Move the item at the selected coordinate into the agent's inventory
        if action.type == ActionType.PICKUP:
            assert isinstance(action.argument, PickupArgument), "Incompatible action and argument type"
            coord = action.argument.coord
            # Only one item allowed in inventory
            if len(next_state.inventory) == 0:
                room = next_state.get_current_room()
                # Convert the agent-relative coordinate to the room coordinate
                coord = config.agent_to_room_item_coord(coord)
                item = room.pop_item(coord)
                if item is not None:
                    next_state.place_in_inventory(item)

        # Move the item in the agent's inventory to the selected coordinate, so long as it doesn't overlap anoterh item
        if action.type == ActionType.DROP:
            assert isinstance(action.argument, PickupArgument), "Incompatible action and argument type"
            coord = action.argument.coord
            room = next_state.get_current_room()
            if len(next_state.inventory) > 0:
                item = copy.deepcopy(next_state.inventory[0])
                coord = config.agent_to_room_item_coord(coord)
                item.coord = coord # Update item coordinates to where it's about to be placed
                result = room.push_item(item)
                if result: # If successfuly placed, remove from inventory
                    next_state.inventory = next_state.inventory[1:]

        return next_state

    def get_env_state(self):
        world = copy.deepcopy(self.world)
        task = copy.deepcopy(self.task)
        return {"world": world, "task": task, "steps": self.steps}

    def set_env_state(self, state):
        self.world = copy.deepcopy(state["world"])
        self.task = copy.deepcopy(state["task"])
        self.steps = state["steps"]

    def set_horizon(self, horizon):
        self.horizon = horizon

    def reset(self) -> (BwObservation, BwTask):
        oracle_agent = OracleAgent()
        while True:
            self.world = World.make_random()
            self.task = generate_random_task()
            self.steps = 0
            observation = self.world.get_observation(config.FULL_OBSERVABILITY)
            observation.privileged_info.attach_task(self.task)

            # Check that the task can be completed - oracle agent just stops immediately if the task is not possible.
            # Also, check that the task is not already completed.
            oracle_agent.start_new_rollout(self.task)
            if oracle_agent.act(observation).type != ActionType.STOP and not self.task.check_goal_conditions(self.world):
                break

        if self.device:
            observation = observation.to(self.device)

        return observation, self.task

    def step(self, action: Action) -> Tuple[BwObservation, float, bool, Dict]:
        next_state = self._transition_function(self.world, action)
        reward = self.task.compute_reward(self.world, next_state)
        task_success = self.task.check_goal_conditions(next_state)
        stopped = self.world.stopped

        # Terminate episode if stop action was executed or if max steps exceeded
        done = stopped or self.steps >= self.horizon

        observation = next_state.get_observation(config.FULL_OBSERVABILITY)
        observation.privileged_info.attach_task(self.task)
        if self.device:
            observation = observation.to(self.device)

        self.world = next_state
        self.steps += 1

        md = {
            "success": task_success
        }
        return observation, reward, done, md
