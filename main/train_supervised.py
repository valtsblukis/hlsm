"""
Quick experiment for training spatial dynamics model in the blockworld environment in a behavior cloning setting,
following Oracle trajectories
"""
from typing import Dict
import os
import sys
import torch
import torch.multiprocessing as mp

from main.data_loading import make_disk_dataloaders, make_navigation_dataloaders, make_perception_dataloaders

from main.train_loop import train_eval_loop
from lgp.utils.better_summary_writer import BetterSummaryWriter
from lgp.parameters import Hyperparams, load_experiment_definition
from lgp import model_registry
from lgp import paths


def resolve_model_path(filename):
    models_dir = paths.get_model_dir()
    os.makedirs(models_dir, exist_ok=True)
    file_path = os.path.join(models_dir, filename)
    return file_path


def resolve_checkpoint_path(filename):
    checkpoints_dir = paths.get_checkpoint_dir()
    os.makedirs(checkpoints_dir, exist_ok=True)
    file_path = os.path.join(checkpoints_dir, filename)
    return file_path


def save_checkpoint(model, optimizers, model_file, checkpoint_file, epoch, iter):
    state_dict = model.state_dict()
    save_model_file_last = f"{model_file}.pytorch"
    save_model_file_e = f"{model_file}_e{epoch}.pytorch"
    torch.save(state_dict, resolve_model_path(save_model_file_last))
    torch.save(state_dict, resolve_model_path(save_model_file_e))
    print(f"Saved trained model in: {save_model_file_e} and {save_model_file_last}")

    checkpoint = {
        "nonbert_optimizer": optimizers[0].state_dict(),
        "bert_optimizer": optimizers[1].state_dict() if optimizers[1] is not None else None,
        "epoch": epoch,
        "iter": iter
    }
    # Saving two files guarantees that if we run out of space, at least one of them will be a valid checkpoint.
    # and allows recovering training
    torch.save(checkpoint, resolve_checkpoint_path(checkpoint_file))
    torch.save(checkpoint, resolve_checkpoint_path(f"{checkpoint_file}_b"))
    print(f"Saved checkpoint in: {checkpoint_file} and {checkpoint_file}_b")


def load_checkpoint(checkpoint_file, model_file):
    model_state = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    nonbert_optimizer_state = checkpoint["nonbert_optimizer"]
    bert_optimizer_state = checkpoint["bert_optimizer"]
    epoch = checkpoint["epoch"]
    iter = checkpoint["iter"]
    return (nonbert_optimizer_state, bert_optimizer_state), model_state, epoch, iter


def train_main(exp_def: Dict):
    setup = exp_def.get("Setup")

    env_type = setup.get("env")
    env_setup = setup.get("env_setup")
    device = torch.device(setup.get("device", "cpu"))
    exp_name = setup.get("experiment_name")
    max_rollouts = setup.get("max_rollouts")
    load_model_file = setup.get("load_model_file", None)
    save_model_file = setup.get("save_model_file")
    save_checkpoint_file = setup.get("save_checkpoint_file")
    load_checkpoint_file = setup.get("load_checkpoint_file")

    num_epochs = setup.get("num_epochs")
    model_type = setup.get("model_type")
    batch_size = setup.get("batch_size", 1)
    dataset_type = setup.get("dataset_type", "subgoals")
    num_workers = setup.get("num_workers", 0)

    hyperparams = Hyperparams(exp_def.get("Hyperparams"))
    print(f"Loading dataset")
    if dataset_type == "subgoals":
        train_loader, val_loader = make_disk_dataloaders("tapm", env_setup, max_rollouts, env_type, hyperparams, batch_size, num_workers)
    elif dataset_type == "navigation":
        train_loader, val_loader = make_navigation_dataloaders(env_setup, max_rollouts, batch_size, num_workers)
    elif dataset_type == "perception":
        train_loader, val_loader = make_perception_dataloaders(env_setup, max_rollouts, batch_size, num_workers)
    else:
        raise ValueError(f"Unknown dataloder type: {dataset_type}")

    # Tensorboard
    writer = BetterSummaryWriter(paths.get_experiment_runs_dir(exp_name), start_iter=0)

    # Model
    print(f"Loading model: {model_type}")
    model = model_registry.get_model(model_type)(hyperparams)
    print(f"Total # parameters: {sum(p.numel() for p in model.parameters())}")

    # Continue from a checkpoint
    if load_checkpoint_file is not None:
        print(f"LOADING CHECKPOINT FROM: {load_checkpoint_file} WITH MODEL FROM {load_model_file}")
        optimizers = None
        optimizer_states, model_state, start_epoch, start_iter = load_checkpoint(load_checkpoint_file, load_model_file)
        model.load_state_dict(model_state)
    else:
        optimizers = None
        optimizer_states = None
        start_epoch = 0
        start_iter = 0
        if load_model_file is not None:
            print(f"LOADING MODEL FROM: {load_model_file}")
            # I think this is initializing the cuda context?
            model.load_state_dict(torch.load(load_model_file))

    model = model.to(device)

    gstep = start_iter  # Number of iterations of training
    for i in range(start_epoch, num_epochs, 1):
        gstep, optimizers = train_eval_loop(train_loader,
                                            model, writer,
                                            val=False,
                                            optimargs=hyperparams.get("optimizer_args").d,
                                            gstep=gstep,
                                            device=device,
                                            optimizers=optimizers,
                                            optimizer_states=optimizer_states)

        train_eval_loop(val_loader,
                        model,
                        writer,
                        val=True,
                        optimargs=hyperparams.get("optimizer_args").d,
                        device=device)

        save_checkpoint(model, optimizers, save_model_file, save_checkpoint_file, i, gstep)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    def_name = sys.argv[1]
    torch.cuda.synchronize()
    exp_def = load_experiment_definition(def_name)
    setup = exp_def.get("Setup")
    train_main(exp_def)
