import os
import copy

from typing import Tuple, Dict, Iterator, Union, Collection

from lgp.abcd.env import Env

from alfred.env.thor_env import ThorEnv

from lgp.env.alfred.state_tracker import StateTracker
from lgp.env.alfred.wrapping.args import get_faux_args

from lgp.env.alfred.tasks import AlfredTask, TaskRecord
from lgp.env.alfred.alfred_observation import AlfredObservation
from lgp.env.alfred.alfred_action import AlfredAction

from lgp.env.alfred import config
from lgp.utils.utils import SimpleProfiler

PROFILE = False

DEFAULT_SETUP = {
    "data_splits": ["train"],
    "filter_task_types": [],
    "no_segmentation": False,
    "no_depth": False,
    "max_fails": 10
}


class AlfredEnv(Env):

    def __init__(self, device=None, setup=None, hparams=None):
        super().__init__()
        alfred_display = (os.environ.get("ALFRED_DISPLAY")
                          if "ALFRED_DISPLAY" in os.environ
                          else os.environ.get("DISPLAY"))
        if alfred_display.startswith(":"):
            alfred_display = alfred_display[1:]

        self.thor_env = ThorEnv(x_display=alfred_display)
        self.task = None
        self.steps = 0
        self.device = device
        self.horizon : int = config.DEFAULT_HORIZON
        self.fail_count : int = 0

        if not setup:
            self.setup = DEFAULT_SETUP
        else:
            self.setup = setup

        self.data_splits = self.setup["data_splits"]
        # Optionally filter tasks by type
        allowed_tasks = self.setup["filter_task_types"]
        allowed_ids = self.setup.get("filter_task_ids", None)

        # Setup state tracker
        reference_seg = self.setup.get("reference_segmentation", False)
        reference_depth = self.setup.get("reference_depth", False)
        reference_inventory = self.setup.get("reference_inventory", False)
        reference_pose = self.setup.get("reference_pose", False)
        print(f"USING {'REFERENCE DEPTH' if reference_depth else 'PREDICTED DEPTH'} "
              f"and {'REFERENCE SEGMENTATION' if reference_seg else 'PREDICTED SEGMENTATION'}")

        self.max_fails = setup.get("max_fails", 10)
        print(f"Max failures: {self.max_fails}")
        self.state_tracker = StateTracker(reference_seg=reference_seg,
                                          reference_depth=reference_depth,
                                          reference_inventory=reference_inventory,
                                          reference_pose=reference_pose,
                                          hparams=hparams)

        if allowed_tasks is not None:
            print(f"FILTERING TASKS: {allowed_tasks}")
            task_filter = AlfredTask.make_task_type_filter(allowed_tasks)
        elif allowed_ids is not None:
            print(f"FILTERING TASKS: {allowed_ids}")
            task_filter = AlfredTask.make_task_id_filter(allowed_ids)
        else:
            raise ValueError("")
        self.task_iterator = AlfredTask.iterate_all_tasks(data_splits=self.data_splits, task_filter=task_filter)
        self.reward_type = self.setup.get("reward_type", "sparse")
        self.smooth_nav = self.setup.get("smooth_nav", False)
        self.task_num_range = None

        self.prof = SimpleProfiler(print=PROFILE)

    def get_env_state(self) -> Dict:
        world = copy.deepcopy(self.world)
        task = copy.deepcopy(self.task)
        return {"world": world, "task": task, "steps": self.steps}

    def set_env_state(self, state: Dict):
        self.world = copy.deepcopy(state["world"])
        self.task = copy.deepcopy(state["task"])
        self.steps = state["steps"]

    def set_horizon(self, horizon):
        self.horizon = horizon

    def set_task_iterator(self, task_iterator: Union[Iterator, None]):
        self.task_iterator = task_iterator

    def set_task_num_range(self, task_num_range):
        # Setting this to different ranges on different processes allows parallelizing the data collection effort
        if task_num_range is not None:
            self.task_num_range = list(task_num_range)

    def _choose_task(self):
        assert self.task_iterator is not None, "Missing task iterator"
        try:
            while True:
                task, i = next(self.task_iterator)
                if self.task_num_range is None or i in self.task_num_range:
                    return task, i
        except StopIteration as e:
            raise StopIteration

    def reset(self, randomize=False, skip_tasks: Union[Collection[TaskRecord], None] = None) -> (AlfredObservation, AlfredTask):
        self.task, task_number = self._choose_task()

        # Skip tasks that are already completed
        task_id = self.task.get_task_id()
        repeat_idx = self.task.get_repeat_idx()
        if skip_tasks is not None:
            if self.task.get_record() in skip_tasks:
                print(f"Skipping task: {task_id} : {repeat_idx}")
                return None, None, None
            else:
                print(f"Including task: {task_id} : {repeat_idx}")

        self.fail_count = 0
        self.steps = 0

        # Apply patch to shift "LookDown" actions to right before interaction, and add four explore actions
        if not self.task.traj_data.is_test():
            self.task.traj_data.patch_trajectory()

        object_poses = self.task.traj_data.get_object_poses()
        dirty_and_empty = self.task.traj_data.get_dirty_and_empty()
        object_toggles = self.task.traj_data.get_object_toggles()

        self.prof.tick("proc")
        # Resetting alfred.env.thor_env.ThorEnv
        # see alfred/models/eval/eval.py:setup_scene (line 100) for reference
        self.thor_env.reset(self.task.traj_data.get_scene_number(),
                                  render_image=False,
                                  render_depth_image=True,
                                  render_class_image=False,
                                  render_object_image=True)
        self.thor_env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        _ = self.thor_env.step(self.task.traj_data.get_init_action(), smooth_nav=self.smooth_nav)

        # If this is not a test example, estting the task here allows tracking results (e.g. goal-conditions)
        if not self.task.traj_data.is_test():
            # The only argument in args that ThorEnv uses is args.reward_config, which is kept to its default
            self.thor_env.set_task(self.task.traj_data.data, get_faux_args(), reward_type=self.reward_type)
        self.prof.tick("thor_env_reset")
        print(f"Task: {str(self.task)}")
        event = self.thor_env.last_event
        self.state_tracker.reset(event)
        observation = self.state_tracker.get_observation()

        if self.device:
            observation = observation.to(self.device)

        return observation, self.task, task_number

    def _error_is_fatal(self, err):
        self.fail_count += 1
        if self.fail_count >= self.max_fails:
            print(f"EXCEEDED MAXIMUM NUMBER OF FAILURES ({self.max_fails})")
            return True
        else:
            return False

    def step(self, action: AlfredAction) -> Tuple[AlfredObservation, float, bool, Dict]:
        self.prof.tick("out")

        # The ALFRED API does not accept the Stop action, do nothing
        message = ""
        if action.is_stop():
            done = True
            transition_reward = 0
            api_action = None
            events = []

        # Execute all other actions in the ALFRED API
        else:
            alfred_action, interact_mask = action.to_alfred_api()
            self.prof.tick("proc")

            ret = self.thor_env.va_interact(
                alfred_action,
                interact_mask,
                smooth_nav=self.smooth_nav)

            # Default version of ALFRED
            if len(ret) == 5:
                exec_success, event, target_instance_id, err, api_action = ret
                events = []
            # Patched version of ALFRED that returns intermediate events from smooth actions
            # To use this, apply the patch alfred-patch.patch onto the ALFRED code:
            # $ git am alfred-patch.patch
            elif len(ret) == 6:
                exec_success, event, events, target_instance_id, err, api_action = ret
            else:
                raise ValueError("Invalid number of return values from ThorEnv")

            self.prof.tick("thor_env_interact")
            if not self.task.traj_data.is_test():
                transition_reward, done = self.thor_env.get_transition_reward()
                done = False
            else:
                transition_reward, done = 0, False

            if not exec_success:
                fatal = self._error_is_fatal(err)
                print(f"ThorEnv {'fatal' if fatal else 'non-fatal'} Exec Error: {err}")
                if fatal:
                    done = True
                    api_action = None
                message = str(err)

        self.prof.tick("step")

        # Track state (pose and inventory) from RGB images and actions
        event = self.thor_env.last_event
        self.state_tracker.log_action(action)
        self.state_tracker.log_event(event)
        self.state_tracker.log_extra_events(events)

        observation = self.state_tracker.get_observation()
        observation.privileged_info.attach_task(self.task) # TODO: See if we can get rid of this?
        if self.device:
            observation = observation.to(self.device)

        # Rewards and progress tracking metadata
        if not self.task.traj_data.is_test():
            reward = transition_reward - 0.05
            goal_satisfied = self.thor_env.get_goal_satisfied()
            goal_conditions_met = self.thor_env.get_goal_conditions_met()
            task_success = goal_satisfied
            md = {
                "success": task_success,
                "goal_satisfied": goal_satisfied,
                "goal_conditions_met": goal_conditions_met,
                "message": message,
            }
        else:
            reward = 0
            md = {}

        # This is used to generate leaderboard replay traces:
        md["api_action"] = api_action

        self.steps += 1

        self.prof.tick("proc")
        self.prof.loop()
        self.prof.print_stats(20)
        return observation, reward, done, md
