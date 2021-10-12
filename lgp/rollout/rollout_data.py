import os
import torch
import inspect
from typing import Dict, List
import functools
import compress_pickle as pickle


EXT = "gz"
J = 1

# From 0 to 9. Lower numbers generally give faster, but less efficient compression.
# In my experiments, I found that compresslevel=2 actually was fastest, but 50% less space efficient than compresslevel=9
COMPRESSLEVEL = 1

#prof = SimpleProfiler(print=True)


def rollouts_to_device(x, device):
    # Convert tuples to lists
    if isinstance(x, tuple):
        x = [y for y in x]
    # Iterate lists
    if isinstance(x, list):
        for i,y in enumerate(x):
            x[i] = rollouts_to_device(y, device)
    # Iterate dicts
    if isinstance(x, dict):
        for k,v in x.items():
            x[k] = rollouts_to_device(v, device)
    # Move to the desired device
    if hasattr(x, "to") and callable(x.to):
        with torch.no_grad():
            x = x.to(device)
    return x


# TODO: Agree on a convention to treat the data and standardize
def load_rollout(dataset_path: str, num: int) -> List[Dict]:
    path = os.path.join(dataset_path, f"rollout_{num}.{EXT}")
    rollout = pickle.load(path)
    rollout = rollouts_to_device(rollout, "cpu")
    return rollout


def load_rollout_from_path(rollout_path) -> List[Dict]:
    rollout = pickle.load(rollout_path)
    # rollout = rollouts_to_device(rollout, "cpu")
    return rollout


def save_rollout_to_path(rollout, rollout_path):
    pickle.dump(rollout, f"{rollout_path}.{EXT}", compresslevel=COMPRESSLEVEL)


def save_rollout(rollout: List[Dict], dataset_path: str, num: int):
    rollout = rollouts_to_device(rollout, "cpu")
    os.makedirs(dataset_path, exist_ok=True)
    rollout_path = os.path.join(dataset_path, f"rollout_{num}.{EXT}")
    pickle.dump(rollout, rollout_path, compresslevel=COMPRESSLEVEL)


def dump(rollout, rollout_path):
    pickle.dump(rollout, rollout_path, compresslevel=COMPRESSLEVEL)


def load(rollout_path):
    return pickle.load(rollout_path)


def rollout_exists(dataset_path: str, num: int):
    os.makedirs(dataset_path, exist_ok=True)
    rollout_path = os.path.join(dataset_path, f"rollout_{num}.{EXT}")
    return os.path.exists(rollout_path)


def save_rollouts(data: List[List[Dict]], dataset_path: str):
    os.makedirs(dataset_path, exist_ok=True)
    for i, rollout in enumerate(data):
        save_rollout(rollout, dataset_path, i)


def list_rollouts(dataset_path: str) -> List[int]:
    rollout_paths = os.listdir(dataset_path)
    rollout_paths = [r for r in rollout_paths if r.endswith(EXT) and "rollout" in r]
    rollout_numbers = list(sorted([int(r.split("_")[1].split(".")[0]) for r in rollout_paths]))
    return rollout_numbers


def load_rollouts(dataset_path: str, max_count: int = -1) -> List[List[Dict]]:
    rollout_numbers = list_rollouts(dataset_path)
    if max_count > 0:
        rollout_numbers = rollout_numbers[:max_count]
    if J == 1:
        rollouts = [load_rollout(dataset_path, i) for i in rollout_numbers]
    else:
        ctx = mp.get_context('spawn')
        p = ctx.Pool(J)
        rollouts = p.map(functools.partial(load_rollout, dataset_path), rollout_numbers)
        p.close()
        p.join()
    return rollouts


# Temp code to move all rollouts to CPU:

import pathlib
import datetime
import gzip


def convert_path(arg):
    i, rollout_path = arg
    print(f"Converting: {i} : {rollout_path}")
    print("             Loading")
    try:
        rollout = pickle.load(rollout_path)
        rollout = rollouts_to_device(rollout, "cpu")
        pickle.dump(rollout, rollout_path)
        print("             Saving")
    except (EOFError, gzip.BadGzipFile) as e:
        print(f"Deleting corrupted file: {rollout_path}")
        os.remove(rollout_path)


def filter_older(paths, than_date):
    out_paths = []
    for path in paths:
        fname = pathlib.Path(path)
        ftime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
        delta = ftime - than_date
        if delta.total_seconds() < 0:
            out_paths.append(path)
    return out_paths


if __name__ == "__main__":

    dataset_dir = "/media/valts/Data/lgdts/data/rollouts/navigation_data"
    dataset_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    older_than = datetime.datetime(year=2021, month=4, day=22, hour=6, minute=0, second=0, microsecond=0)
    old_dataset_paths = filter_older(dataset_paths, older_than)
    print(f"{len(old_dataset_paths)} / {len(dataset_paths)} files are older than {older_than}")
    old_dataset_paths = dataset_paths

    old_dataset_paths = [(i, f) for i, f in enumerate(old_dataset_paths)]

    import multiprocessing as mp
    mp.set_start_method("spawn")
    # Activate CUDA context
    import torch
    torch.cuda.synchronize()

    with mp.Pool(6) as p:
        p.map(convert_path, old_dataset_paths)

