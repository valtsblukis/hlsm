"""
Script that rolls out an agent and does not much else for now
"""
import sys
import os
import torch


from lgp.agents.agents import get_agent
from lgp.rollout.rollout_actor import RolloutActorLocal
from lgp.metrics.alfred_eval import get_multiple_rollout_metrics_alfred
from main.visualize_rollout import visualize_rollout
from lgp.parameters import Hyperparams, load_experiment_definition

from main.eval_progress import EvalProgress

from lgp.env.alfred.alfred_env import AlfredEnv


def evaluate_rollouts(exp_def, rollouts):
    metrics = get_multiple_rollout_metrics_alfred(rollouts)
    print("Results: ")
    metrics.printout()


def collect_rollouts(exp_def):
    # params
    device = torch.device(exp_def.Setup.device)
    dataset_device = torch.device(exp_def.Setup.dataset_device)
    exp_name = exp_def.Setup.experiment_name
    horizon = exp_def.Setup.horizon
    num_rollouts = exp_def.Setup.num_rollouts
    visualize_rollouts = exp_def.Setup.visualize_rollouts
    save_animation_dir = exp_def.Setup.get("save_rollout_animations_dir", False)

    env = AlfredEnv(device=device,
                    setup=exp_def.Setup.env_setup.d,
                    hparams=exp_def.Hyperparams.d)

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

    # Track progress
    eval_progress = EvalProgress(exp_name)

    # Collect the rollouts
    for i in range(num_rollouts):
        print(f"Rollout {i}/{num_rollouts}")
        try:
            rollout = rollout_actor.rollout(skip_tasks=eval_progress.get_done_tasks())

            if rollout is not None:
                eval_progress.add_rollout(rollout)

                if save_animation_dir is not None:
                    for s in rollout:
                        if s["observation"] is not None:
                            s["observation"].compress()
                    os.makedirs(save_animation_dir, exist_ok=True)
                    visualize_rollout(rollout, save_animation_dir, f"rollout_{i}", start_t=0)

                eval_progress.save()
        except StopIteration as e:
            break

    # Export the leaderboard file
    eval_progress.export_leaderboard_json()

    # Get numbers for table on validation sets
    evaluate_rollouts(exp_def, eval_progress.get_rollout_list())


if __name__ == "__main__":
    def_name = sys.argv[1]
    exp_def = Hyperparams(load_experiment_definition(def_name))
    collect_rollouts(exp_def)