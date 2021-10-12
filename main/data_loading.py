import os

import torch
from torch.utils.data import DataLoader

from lgp import paths
from lgp.datasets.tapm_dataset import TapmDataset
from lgp.datasets.navigation_dataset import NavigationDataset
from lgp.datasets.perception_dataset import PerceptionDataset

from lgp.models.alfred.hlsm.hlsm_model_factory import HlsmModelFactory


def get_model_factory(env_type, hparams):
    if env_type == "Alfred":
        print("ALFRED enviornment")
        model_factory = HlsmModelFactory(hparams)
    else:
        raise ValueError(f"Unrecognized environment: {env_type}")
    return model_factory


def split_data(data, train_faction=0.8):
    train_data = data[:int(len(data) * train_faction)]
    val_data = data[len(train_data):]
    return train_data, val_data


def make_dataloader(data_or_paths, dataset_mode, cls, env_type, hparams, batch_size=64, num_workers=0):
    model_factory = get_model_factory(env_type, hparams)

    dataset = cls(data_or_paths,
                  dataset_mode=dataset_mode,
                  model_factory=model_factory,
                  gamma=hparams.gamma)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=dataset.collate_fn,
                            pin_memory=False,
                            drop_last=True)
    return dataloader


def get_incl_task_indices(env_setup):
    from lgp.env.alfred.tasks import AlfredTask
    data_splits = env_setup['data_splits']
    allowed_types = env_setup['filter_task_types']
    include_rollout_indices = []
    if len(allowed_types) > 0:
        task_filter = AlfredTask.make_task_type_filter(allowed_types)
    else:
        task_filter = lambda m: True
    for i, (alfred_task, task_id) in enumerate(AlfredTask.iterate_all_tasks(data_splits, task_filter=task_filter)):
        include_rollout_indices.append(task_id)
    return include_rollout_indices


def filter_rollout_paths(dataset_paths, incl_indices):
    # eval all rollouts that end with "i.gz" where i is an integer among include_rollout_indices
    filtered_paths = [f for f in dataset_paths if int(os.path.basename(f).split("_")[1].split(".")[0]) in incl_indices]
    return filtered_paths


def make_disk_dataloaders(dataset_mode, env_setup, max_rollouts, env_type, hparams, batch_size, num_workers):
    dataset_dir = paths.get_default_subgoal_rollout_data_dir()
    dataset_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    dataset_paths = list(sorted(dataset_paths))
    if max_rollouts > 0:
        dataset_paths = dataset_paths[:max_rollouts]
    train_paths = [p for i, p in enumerate(dataset_paths) if i % 100 != 0]
    val_paths = [p for i, p in enumerate(dataset_paths) if i % 100 == 0]

    incl_rollout_indices = get_incl_task_indices(env_setup)
    train_paths = filter_rollout_paths(train_paths, incl_rollout_indices)
    val_paths = filter_rollout_paths(val_paths, incl_rollout_indices)

    train_loader = make_dataloader(train_paths, dataset_mode, TapmDataset, env_type, hparams, batch_size, num_workers)
    val_loader = make_dataloader(val_paths, dataset_mode, TapmDataset, env_type, hparams, batch_size, num_workers)
    return train_loader, val_loader


def make_navigation_dataloader(data_paths, batch_size, num_workers):
    dataset = NavigationDataset(data_paths)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=dataset.collate_fn,
                            sampler=None,
                            pin_memory=False)
    return dataloader


def make_navigation_dataloaders(env_setup, max_rollouts, batch_size, num_workers):
    dataset_dir = paths.get_default_navigation_rollout_data_dir()
    dataset_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    dataset_paths = list(sorted(dataset_paths))
    if max_rollouts is not None and max_rollouts > 0:
        dataset_paths = dataset_paths[:max_rollouts]
    train_paths = [p for i, p in enumerate(dataset_paths) if i % 100 != 0]
    val_paths = [p for i, p in enumerate(dataset_paths) if i % 100 == 0]

    train_loader = make_navigation_dataloader(train_paths, batch_size, num_workers)
    val_loader = make_navigation_dataloader(val_paths, batch_size, num_workers)
    return train_loader, val_loader


def make_perception_dataloader(data_paths, batch_size, num_workers):
    dataset = PerceptionDataset(data_paths)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=dataset.collate_fn,
                            sampler=None,
                            pin_memory=False)
    return dataloader


def make_perception_dataloaders(env_setup, max_rollouts, batch_size, num_workers):
    dataset_dir = paths.get_default_navigation_rollout_data_dir()
    dataset_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    dataset_paths = list(sorted(dataset_paths))
    if max_rollouts is not None and max_rollouts > 0:
        dataset_paths = dataset_paths[:max_rollouts]
    train_paths = [p for i, p in enumerate(dataset_paths) if i % 100 != 0]
    val_paths = [p for i, p in enumerate(dataset_paths) if i % 100 == 0]

    train_loader = make_perception_dataloader(train_paths, batch_size, num_workers)
    val_loader = make_perception_dataloader(val_paths, batch_size, num_workers)
    return train_loader, val_loader