"""
Script that rolls out an agent and does not much else for now
"""

import sys
import os
import torch
import json
import datetime

from lgp.agents.agents import get_agent
from lgp.rollout.rollout_actor import RolloutActorLocal
from lgp.metrics.alfred_eval import get_multiple_rollout_metrics_alfred
import lgp.rollout.rollout_data as rd
from main.visualize_rollout import visualize_rollout
from lgp.parameters import Hyperparams, load_experiment_definition

from lgp.env.alfred.alfred_env import AlfredEnv


def evaluate_rollouts(exp_def, rollouts):
    metrics = get_multiple_rollout_metrics_alfred(rollouts)
    print("Results: ")
    metrics.printout()


class LeaderboardProgress:
    def __init__(self, exp_def):
        self.progress_file = exp_def.Setup.leaderboard_progress_file
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as fp:
                self.progress = json.load(fp)
                print(f"Continue where left off? {len(self.progress)} rollouts completed!")
                inp = input("y/n >")
                if inp != "y":
                    print("Stopping")
                    sys.exit(0)
                # Convert old format with only the first repeat to new format indexed by repeats:
                if len(self.progress[0]) == 3:
                    print("Converting progress representation to new format with repeat_idx")
                    self.progress = [[x[0], 0, x[1], x[2]] for x in self.progress]
        else:
            self.progress = []

        self.tasks_done = [(p[0], p[1]) for p in self.progress]

    def did_already_collect(self, task_id, repeat_idx):
        return (task_id, repeat_idx) in self.tasks_done

    def add_rollout(self, rollout):
        task_id = rollout[0]["task"].get_task_id()
        repeat_idx = rollout[0]["task"].get_repeat_idx()
        datasplit = rollout[0]["task"].get_data_split()
        actseq = [s["md"]["api_action"] for s in rollout if s["md"]["api_action"] is not None]
        tagged_actseq = [task_id, repeat_idx, datasplit, actseq]
        self.progress.append(tagged_actseq)
        self.tasks_done.append((task_id, repeat_idx))

    def save_json(self):
        with open(f"{self.progress_file}_b", "w") as fp:
            json.dump(self.progress, fp, indent=4, sort_keys=True)
        with open(self.progress_file, "w") as fp:
            json.dump(self.progress, fp, indent=4, sort_keys=True)
        print(f"Saved progress in {self.progress_file}")

    def export_leaderboard_json(self):
        leaderboard_json = {
        }
        for task_id, repeat_idx, datasplit, actseq in self.progress:
            if datasplit not in leaderboard_json:
                leaderboard_json[datasplit] = []
            leaderboard_json[datasplit].append({task_id: actseq})
        leaderboard_dir = os.path.dirname(self.progress_file)
        leaderboard_file = os.path.join(leaderboard_dir, 'tests_actseqs_dump_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(leaderboard_file, "w") as fp:
            json.dump(leaderboard_json, fp, indent=4, sort_keys=True)
            print(f"Dumped action sequences to: {leaderboard_file}")


def collect_rollouts(exp_def):
    # params
    device = torch.device(exp_def.Setup.device)
    dataset_device = torch.device(exp_def.Setup.dataset_device)
    exp_name = exp_def.Setup.experiment_name
    horizon = exp_def.Setup.horizon
    num_rollouts = exp_def.Setup.num_rollouts
    visualize_rollouts = exp_def.Setup.visualize_rollouts
    tasks_done = None

    save_animation_dir = exp_def.Setup.get("save_rollout_animations_dir", False)
    load_rollouts_dir = exp_def.Setup.get("load_rollouts_dir", False)
    save_rollouts_dir = exp_def.Setup.get("save_rollouts_dir", False)

    env = AlfredEnv(device=device, setup=exp_def.Setup.env_setup.d, hparams=exp_def.Hyperparams.d)

    agent = get_agent(exp_def.Setup, exp_def.Hyperparams, device)

    rollout_actor = RolloutActorLocal(experiment_name=exp_name,
                                      agent=agent,
                                      env=env,
                                      dataset_proc=None,
                                      param_server_proc=None,
                                      max_horizon=horizon,
                                      dataset_device=dataset_device,
                                      index=1,
                                      collect_trace=visualize_rollouts,
                                      lightweight_mode=not visualize_rollouts)

    if exp_def.Setup.leaderboard_progress_file:
        leaderboard_progress = LeaderboardProgress(exp_def)
    else:
        print("NOT COLLECTING LEADERBOARD TRACES!")
        leaderboard_progress = None

    # Load previously saved rollouts # TODO: Delete this code
    if load_rollouts_dir:
        rollouts = rd.load_rollouts(load_rollouts_dir)
        tasks_done = [(r[0]["task"].get_task_id(), r[0]["task"].get_repeat_idx()) for r in rollouts]
        print(f"Loaded {len(rollouts)} rollouts from: {load_rollouts_dir}")
    else:
        rollouts = []

    # Save rollouts for later analysis
    if save_rollouts_dir:
        os.makedirs(save_rollouts_dir, exist_ok=True)

    # Collect the rollouts
    for i in range(num_rollouts):
        print(f"Rollout {i}/{num_rollouts}")
        try:
            ret = None
            done = False
            while not done:
                rollout, ret, done = rollout_actor.split_rollout(skip_tasks=leaderboard_progress.tasks_done if leaderboard_progress else tasks_done, max_section=20, ret=None)

            if rollout is not None:
                for s in rollout:
                    if s["observation"] is not None:
                        s["observation"].compress()
                if leaderboard_progress:
                    leaderboard_progress.add_rollout(rollout)
                    leaderboard_progress.save_json()
                rollouts.append(rollout)

                if save_rollouts_dir:
                    rd.dump(rollout, os.path.join(save_rollouts_dir, f"rollout_{i}.gz"))
                    print(f"Saved rollout to: {save_rollouts_dir}")
        except StopIteration as e:
            break

    # Export the leaderboard file if we're collecting test results
    if leaderboard_progress:
        leaderboard_progress.export_leaderboard_json()

    # Get numbers for table
    evaluate_rollouts(exp_def, rollouts)

    # Save animations
    if visualize_rollouts and save_animation_dir:
        for i, rollout in enumerate(rollouts):
            os.makedirs(save_animation_dir, exist_ok=True)
            visualize_rollout(rollouts[i], save_animation_dir, f"rollout_{i}")


if __name__ == "__main__":
    def_name = sys.argv[1]
    exp_def = Hyperparams(load_experiment_definition(def_name))
    collect_rollouts(exp_def)