"""
Script that collects and saves a long list of {state, action, next_state} dicts.
"""
from typing import List, Dict
import os
import sys
import torch
import json

#from lgp.abcd.functions.action_repr_function import ActionReprFunction


from main.data_collection_strategies.alfred_navigation_chunking_strategy import AlfredNavigationPreproc, NavToGoalChunkingStrategy
from main.data_collection_strategies.alfred_hl_to_ll_chunking_strategy import AlfredHLPreproc, AlfredHLChunkingStrategy

from lgp.models.alfred.handcoded_skills.init_skill import InitSkill

from lgp.parameters import Hyperparams, load_experiment_definition
from lgp.utils.utils import SimpleProfiler

from lgp.rollout.rollout_data import rollouts_to_device

from lgp.factory.alfred_factory import AlfredFactory

from lgp.rollout import rollout_data
import lgp.paths


TOTAL_TASKS = 100000
START_TASK = 0
MAX_H = 200
NUM_GPUS = 3

PROFILE = False

#NAV_DATADIR = lgp.paths.get_data_dir()
#HL_DATADIR = NAV_DATADIR
#DIR_PREFIX = "rollouts_full_v3"


def collect_universal_rollouts(exp_def, proc_id):

    # ----------------------------------------------------------------------------------------------------------------
    # Save navigation data to SSD since it produces a LOT of files, but less big in total
    configs = [
        {
            "preprocessor": AlfredNavigationPreproc(),
            "chunker": NavToGoalChunkingStrategy(),
            "dataset_dir": lgp.paths.get_default_navigation_rollout_data_dir(),
            "singles": True,
            "key": "nav",
            "skip_error_rollouts": False
        },
        {
            "preprocessor": AlfredHLPreproc(),
            "chunker": AlfredHLChunkingStrategy(),
            "dataset_dir": lgp.paths.get_default_subgoal_rollout_data_dir(),
            "singles": False,
            "key": "hl",
            "skip_error_rollouts": True
        }
    ]
    progress = {
        "collected_rollouts": [],
        "current_i": START_TASK,
        "chunk_numbers": {
            "hl": 0,
            "nav": 0
        }
    }
    progress_log_file = os.path.join(lgp.paths.get_default_rollout_data_dir(), f"progress_log.json")

    if os.path.exists(progress_log_file):
        print(f"Progress file exists at {progress_log_file}.")
        n = input(f"Continue data collection where left off? (y/n)")
        if n.lower() == "y" or n.lower() == "yes":
            pass
        else:
            print("Aborting data collection. If you don't want to continue where left off, delete the progress file.")
            sys.exit(-1)
        with open(progress_log_file, "r") as fp:
            progress = json.load(fp)

    # ----------------------------------------------------------------------------------------------------------------

    numrange = list(range(progress["current_i"], TOTAL_TASKS))
    device = f"cuda:{proc_id % NUM_GPUS}"

    with torch.no_grad():
        prof = SimpleProfiler(print=PROFILE)
        setup = exp_def.get("Setup")
        setup["device"] = device
        hparams = Hyperparams(exp_def.get("Hyperparams"))

        factory = AlfredFactory()
        env = factory.get_environment(setup, task_num_range=numrange)
        agent = factory.get_agent(Hyperparams(setup), hparams)
        model_factory = factory.get_model_factory(setup, hparams)

        obs_func = model_factory.get_observation_function()

        for i in numrange:
            progress["current_i"] = i
            error_rollout = False

            rollout : List[Dict] = []
            state_repr = None

            try:
                observation, task, task_number = env.reset()
            except StopIteration:
                break

            # Skip already collected rollouts
            if task_number in progress["collected_rollouts"]:
                print(f"Skipping rollout: {task_number} - it exists")
                continue
            else:
                print(f"Collecting rollout: {task_number}")

            agent.start_new_rollout(task)

            for t in range(MAX_H):
                state_repr = obs_func(observation, state_repr, None)
                action = agent.act(observation)
                #action_repr = action_repr_func(action, observation)

                next_observation, reward, done, md = env.step(action)

                if observation.last_action_error:
                    error_rollout = True

                sample = {
                    "task": task,
                    "state_repr": state_repr,
                    "observation": observation,
                    "action": action,
                    #"action_repr": action_repr,
                    "reward": reward,
                    "done": done,
                    "remark": str(agent)
                }
                rollout.append(sample)
                observation = next_observation

                if action.is_stop():
                    break

            # ----------------------------------------------------------------------------------------------------------------

            # Free up GPU memory - do the rest of the stuff on CPU (which is fine)
            rollout = rollouts_to_device(rollout, device="cpu")

            for config in configs:
                movement_stack = []
                chunked_rollout = []
                first = True

                for sample in rollout:
                    movement_stack.append(sample)

                    if config["chunker"].is_sequence_terminal(sample["action"]):
                        # Consider the preceding sequence of navigation actions to be either the result of exploration or
                        # ... part of the manipulation itself
                        chunked_samples = config["chunker"].ll_to_hl(movement_stack, start_idx=InitSkill.sequence_length() if first else 0)
                        if config["chunker"].include_chunk(sample["action"]):
                            chunked_rollout.extend(chunked_samples)
                        movement_stack.clear()
                        first = False

                chunked_rollout = config["preprocessor"].process(chunked_rollout)

                if error_rollout and config["skip_error_rollouts"]:
                    print(f"Skipping saving of rollout: {task_number}")
                else:
                    print(f"Saving rollout: {task_number} of length: {len(rollout)} in {config['dataset_dir']}")
                    if config["singles"]:
                        for sample in chunked_rollout:
                            rollout_data.save_rollout(sample, config["dataset_dir"], progress["chunk_numbers"][config["key"]])
                            progress["chunk_numbers"][config["key"]]+= 1
                    else:
                        rollout_data.save_rollout(chunked_rollout, config["dataset_dir"], task_number)
                        prof.loop()
                        prof.print_stats(1)

            # ----------------------------------------------------------------------------------------------------------------
            #### Save progress
            progress["collected_rollouts"].append(task_number)
            progress["current_i"] = i+1
            with open(progress_log_file, "w") as fp:
                json.dump(progress, fp)


# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    def_name = sys.argv[1]
    exp_def = load_experiment_definition(def_name)
    collect_universal_rollouts(exp_def, 0)
