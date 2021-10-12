"""
This file includes code to index, load, and manage the traj_data json files from ALFRED.
"""
from typing import List, Dict, Union
import os
import json
import copy


# imports from ALFRED
from alfred.gen.utils.image_util import decompress_mask

from lgp.env.alfred.wrapping.paths import get_task_traj_data_path, get_splits_path, get_task_dir_path, get_traj_data_paths


class TrajData:
    def __init__(self, traj_data_path):
        try:
            with open(traj_data_path, "r") as fp:
                data = json.load(fp)
        except json.decoder.JSONDecodeError as e:
            print(f"Couldn't load json: {traj_data_path}")
            raise e
        self.data = data

    def is_test(self):
        # Lazy way to detect test examples - they don't have ground-truth plan data
        return "plan" not in self.data

    def patch_trajectory(self):
        # The ground-truth trajectories are generated with a PDDL planner that has access to ground truth state.
        # as such, it is unconcerned about observability and exploration, and often walks around with it's head
        # down not seeing anything. This is terrible for building voxel map representations of the world.
        # Here we patch the ground truth sequences by making sure that the agent always walks around with it's head
        # tilted down 30 degrees, and then tilts to the correct angle before taking each action.
        # Sometimes this results in invalid action sequences, when the object the agent is holding collides with
        # the environment.
        self.fix_lookdown()
        #self.add_rotate_explore()

    """
    def add_rotate_explore(self):
        proto_rotateleft = {
            "api_action": {
                "action": "RotateLeft",
                "forceAction": True
            },
            "discrete_action": {
                "action": "RotateLeft_90",
                "args": {}
            },
            "high_idx": 0
        }
        self.data["plan"]["low_actions"] = [proto_rotateleft for _ in range(4)] + self.data["plan"]["low_actions"]
    """

    def fix_lookdown(self):
        old_plan = self.data["plan"]["low_actions"]
        plan = copy.deepcopy(old_plan)
        n = len(plan)

        proto_ld = {
            "api_action": {
                "action": "LookDown",
                "forceAction": True
            },
            "discrete_action": {
                "action": "LookDown_15",
                "args": {}
            },
            "high_idx": 0
        }
        proto_lu = {
            "api_action": {
                "action": "LookUp",
                "forceAction": True
            },
            "discrete_action": {
                "action": "LookUp_15",
                "args": {}
            },
            "high_idx": 0
        }

        # First mark for each action (except LookUp, LookDown), how many lookdowns have been done
        step_ldc = []
        ldc = 0
        for i in range(n):
            act_i = plan[i]["api_action"]["action"]
            if act_i == "LookDown":
                ldc += 1
            elif act_i == "LookUp":
                ldc -= 1
            else:
                step_ldc.append(ldc)

        # Then delete all LookDown and LookUp actions
        for i in range(n-1, -1, -1):
            act_i = plan[i]["api_action"]["action"]
            if act_i in ["LookDown", "LookUp"]:
                plan = plan[:i] + plan[i+1:]

        assert len(plan) == len(step_ldc)

        # Then insert the right amount of LookDown and LookUp around interaction actions
        new_plan = []
        for i in range(len(step_ldc)):
            act_i = plan[i]["api_action"]["action"]
            if act_i in ["PickupObject", "PutObject", "SliceObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff"]:
                ld = copy.deepcopy(proto_ld)
                lu = copy.deepcopy(proto_lu)
                ld["high_idx"] = plan[i]["high_idx"]
                lu["high_idx"] = plan[i]["high_idx"]
                for c in range(step_ldc[i]):
                    new_plan.append(ld)
                for c in range(0, -step_ldc[i], 1):
                    new_plan.append(lu)
                new_plan.append(plan[i])
                for c in range(step_ldc[i]):
                    new_plan.append(lu)
                for c in range(0, -step_ldc[i], 1):
                    new_plan.append(ld)
            else:
                new_plan.append(plan[i])

        # Replace the plan
        self.data["plan"]["low_actions"] = new_plan

    def iterate_strings(self):
        # Iterate through task descriptions
        for repeat_idx in range(len(self.data["turk_annotations"]["anns"])):
            task_desc = self.get_task_description(repeat_idx)
            yield task_desc
            step_descs = self.get_step_descriptions(repeat_idx)
            for step_desc in step_descs:
                yield step_desc

    def get_task_id(self): # TODO Type
        return self.data["task_id"]

    def get_task_type(self): # TODO Type
        return self.data["task_type"]

    def get_num_repeats(self):
        return len(self.data["turk_annotations"]["anns"])

    def get_task_description(self, repeat_idx):
        return self.data["turk_annotations"]["anns"][repeat_idx]["task_desc"]

    def get_step_descriptions(self, repeat_idx):
        return self.data["turk_annotations"]["anns"][repeat_idx]["high_descs"]

    def get_scene_number(self):
        return self.data["scene"]["scene_num"]

    def get_object_poses(self):
        return self.data["scene"]["object_poses"]

    def get_dirty_and_empty(self):
        return self.data["scene"]["dirty_and_empty"]

    def get_object_toggles(self):
        return self.data["scene"]["object_toggles"]

    def get_init_action(self):
        return dict(self.data["scene"]["init_action"])

    def get_low_actions(self) -> List:
        return self.data["plan"]["low_actions"]

    def get_api_action_sequence(self) -> Union[List[Dict], None]:
        sequence = self.get_low_actions()
        api_ish_sequence = [{
            "action": a["api_action"]["action"],
            "mask": decompress_mask(a["discrete_action"]["args"]["mask"]) if "mask" in a["discrete_action"]["args"] else None
        } for a in sequence]
        return api_ish_sequence


class AlfredAnnotations():

    def __init__(self):
        self.splits_path = get_splits_path()
        with open(self.splits_path, "r") as fp:
            splits = json.load(fp)
        self.splits = splits

    @classmethod
    def load_traj_data_for_task(cls, data_split: str, task_id: str) -> TrajData:
        traj_data_path = get_task_traj_data_path(data_split, task_id)
        traj_data = TrajData(traj_data_path)
        return traj_data

    @classmethod
    def load_all_traj_data(cls, data_split: str) -> List[TrajData]:
        traj_data_paths = get_traj_data_paths(data_split)
        traj_datas = [TrajData(t) for t in traj_data_paths]
        return traj_datas

    def get_alfred_data_splits(self) -> (List[str], Dict[str, List]):
        """
        Return:
            (list, dict)
            list - list of strings of datasplit names
            dict - dictionary, indexed by datasplit names, containing list of dictionaries of format:
                {"repeat_idx": int, "task": str}
        """
        list_of_splits = list(sorted(self.splits.keys()))
        return list_of_splits, self.splits

    def get_all_task_ids_in_split(self, datasplit: str = "train") -> List[str]:
        assert datasplit in self.splits, f"Datasplit {datasplit} not found in available splits: {self.splits.keys()}"
        task_ids = list(sorted(set([d["task"] for d in self.splits[datasplit]])))
        return task_ids

    def get_num_repeats(self, datasplit: str, task_id: str) -> int:
        traj_data = AlfredAnnotations.load_traj_data_for_task(datasplit, task_id)
        return traj_data.get_num_repeats()
