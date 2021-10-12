import os
import sys

import lgp.rollout.rollout_data as rd

from main.visualize_rollout import dynamic_voxel_viz
from lgp.parameters import Hyperparams, load_experiment_definition


def voxel_viz(rollout):
    dynamic_voxel_viz(rollout)


def viz_collected_rollouts(exp_def):
    rollouts_dir = exp_def.Setup.get("save_rollouts_dir", False)
    rollout_files = os.listdir(rollouts_dir)
    for rollout_file in rollout_files:
        print(f"Loading rollout: {rollout_file}")
        rollout = rd.load(os.path.join(rollouts_dir, rollout_file))
        voxel_viz(rollout)


if __name__ == "__main__":
    def_name = sys.argv[1]
    exp_def = Hyperparams(load_experiment_definition(def_name))
    viz_collected_rollouts(exp_def)