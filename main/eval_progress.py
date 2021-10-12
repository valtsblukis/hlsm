from typing import Collection

import os
import sys
import json
from collections import namedtuple

import lgp.paths
import lgp.rollout.rollout_data as rd

from lgp.env.alfred.tasks import TaskRecord

Progress = namedtuple('Progress', ['datasplit', 'task_id', 'repeat_idx', 'action_seq'])


class EvalProgress:
    def __init__(self, exp_name):
        results_dir = lgp.paths.get_results_dir(exp_name)
        self.exp_name = exp_name

        self.progress = []
        self.rollouts = {}

        if os.path.exists(results_dir):
            print(f"Existing progress found in: {results_dir}")
            print(f"Continue where left off?")
            inp = input("y/n >")
            if inp != "y":
                print("Stopping.")
                sys.exit(0)
            self._load_json()
            self._load_rollouts()
            print(f"Continuing... {len(self.progress)} rollouts completed!")
        else:
            os.makedirs(results_dir, exist_ok=True)

        self.tasks_done = [TaskRecord(p.datasplit, p.task_id, p.repeat_idx) for p in self.progress]

    def _load_json(self):
        progress_file = lgp.paths.get_leaderboard_progress_path(self.exp_name)
        with open(progress_file, "r") as fp:
            self.progress = json.load(fp)
            self.progress = [Progress(*p) for p in self.progress]

    def _save_json(self):
        progress_file = lgp.paths.get_leaderboard_progress_path(self.exp_name)
        progress_out = [list(p) for p in self.progress]
        with open(f"{progress_file}_b", "w") as fp:
            json.dump(progress_out, fp, indent=4, sort_keys=True)
        with open(progress_file, "w") as fp:
            json.dump(progress_out, fp, indent=4, sort_keys=True)

    def _load_rollouts(self):
        rollouts_dir = lgp.paths.get_eval_rollout_dir(self.exp_name)
        files = os.listdir(rollouts_dir)
        for file in files:
            rollout = rd.load_rollout_from_path(os.path.join(rollouts_dir, file))
            self.rollouts[self._task_record_from_rollout(rollout)] = rollout

    def _save_rollouts(self):
        rollouts_dir = lgp.paths.get_eval_rollout_dir(self.exp_name)
        for task_record, rollout in self.rollouts.items():
            rollout_name = f"rollout_{str(task_record)}"
            rollout_path = os.path.join(rollouts_dir, rollout_name)
            if not os.path.exists(rollout_path):
                rd.save_rollout_to_path(rollout, os.path.join(rollouts_dir, rollout_name))

    def _task_record_from_rollout(self, rollout):
        task_id = rollout[0]["task"].get_task_id()
        repeat_idx = rollout[0]["task"].get_repeat_idx()
        datasplit = rollout[0]["task"].get_data_split()
        return TaskRecord(datasplit, task_id, repeat_idx)

    def _actseq_from_rollout(self, rollout):
        task_id = rollout[0]["task"].get_task_id()
        repeat_idx = rollout[0]["task"].get_repeat_idx()
        datasplit = rollout[0]["task"].get_data_split()
        actseq = [s["md"]["api_action"] for s in rollout if s["md"]["api_action"] is not None]
        progress_line = Progress(datasplit, task_id, repeat_idx, actseq)
        return progress_line

    def save(self):
        self._save_json()
        self._save_rollouts()
        print(f"Saved progress in {lgp.paths.get_results_dir(self.exp_name)}")

    def did_already_collect(self, datasplit, task_id, repeat_idx):
        return TaskRecord(datasplit, task_id, repeat_idx) in self.tasks_done

    def get_done_tasks(self) -> Collection[TaskRecord]:
        return self.tasks_done

    def get_rollout_list(self):
        return list(self.rollouts.values())

    def add_rollout(self, rollout):
        progress_line = self._actseq_from_rollout(rollout)
        task = self._task_record_from_rollout(rollout)
        self.progress.append(progress_line)
        self.tasks_done.append(task)
        self.rollouts[task] = rollout

    def export_leaderboard_json(self):
        leaderboard_json = {}
        for prg in self.progress:
            if prg.datasplit not in leaderboard_json:
                leaderboard_json[prg.datasplit] = []
            leaderboard_json[prg.datasplit].append({prg.task_id: prg.action_seq})
        leaderboard_export_file = lgp.paths.get_leaderboard_result_path(self.exp_name)
        with open(leaderboard_export_file, "w") as fp:
            json.dump(leaderboard_json, fp, indent=4, sort_keys=True)
            print(f"Saved leaderboard action sequences to: {leaderboard_export_file}")
            print(f"Upload to ALFRED leaderboard to obtain test results")
