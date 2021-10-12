from typing import List
import os


def get_alfred_root_path():
    assert 'ALFRED_ROOT' in os.environ, "ALFRED_ROOT environment variable not defined!"
    return os.environ['ALFRED_ROOT']


def get_task_traj_data_path(data_split: str, task_id: str) -> str:
    alfred_root = get_alfred_root_path()
    traj_data_path = os.path.join(alfred_root, "data", "json_2.1.0", data_split, task_id, "traj_data.json")
    return traj_data_path


def get_traj_data_paths(data_split: str) -> List[str]:
    alfred_root = get_alfred_root_path()
    traj_data_root = os.path.join(alfred_root, "data", "json_2.1.0", data_split)
    all_tasks = os.listdir(traj_data_root)
    traj_data_paths = []
    for task in all_tasks:
        trials = os.listdir(os.path.join(traj_data_root, task))
        for trial in trials:
            traj_data_paths.append(os.path.join(traj_data_root, task, trial, "traj_data.json"))
    return traj_data_paths


def get_task_dir_path(data_split: str, task_id: str) -> str:
    alfred_root = get_alfred_root_path()
    task_dir_path = os.path.join(alfred_root, "data", "json_2.1.0", data_split, task_id.split("/")[0])
    return task_dir_path


def get_splits_path():
    splits_path = os.path.join(get_alfred_root_path(), "data", "splits", "oct21.json")
    return splits_path