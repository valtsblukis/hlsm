import ray
import torch
from lgp.abcd.agent import TrainableAgent
from lgp import paths


class RolloutActorLocal:

    def __init__(self,
                 experiment_name: str,
                 agent : TrainableAgent,
                 env,
                 dataset_proc,
                 param_server_proc,
                 max_horizon,
                 dataset_device,
                 index,
                 collect_trace=False,
                 lightweight_mode=False):
        self.dataset_process = dataset_proc
        self.param_server_proc = param_server_proc
        self.actor_index = index
        if self.actor_index == 0:
            from lgp.utils.better_summary_writer import BetterSummaryWriter
            self.writer = BetterSummaryWriter(f"{paths.get_experiment_runs_dir(experiment_name)}-rollout", start_iter=0)
        else:
            self.writer = None

        self.agent = agent
        self.env = env
        self.horizon = max_horizon
        self.env.set_horizon(max_horizon)
        self.counter = 0

        self.collect_trace = collect_trace       # Whether to eval outputs of agent.get_trace in the rollout
        self.lightweight_mode = lightweight_mode # Whether to produce stripped-down rollouts with task and metadata only

        self.dataset_device = dataset_device

    def _load_agent_state_from_ps(self):
        for model in self.agent.get_learnable_models():
            model.load_state_dict(ray.get(self.param_server_proc.get.remote(model.get_name())))

    def rollout_and_send_forever(self):
        while True:
            self.rollout_and_send()

    def split_rollout(self, skip_tasks=None, max_section=20, ret=None):
        rollout = []
        with torch.no_grad():
            if ret is None:
                observation, task, rollout_idx = self.env.reset(skip_tasks=skip_tasks)
                # Skipped:
                if task is None:
                    return None, None, True

                print("Task: ", str(task))
                self.agent.start_new_rollout(task)
                action = self.agent.act(observation)
                start = 0
            else:
                observation = ret["observation"]
                action = ret["action"]
                task = ret["task"]
                rollout_idx = ret["rollout_idx"]
                start = ret["t"]

            total_reward = 0
            for t in range(start, self.horizon):
                next_observation, reward, done, md = self.env.step(action)
                total_reward += reward

                rollout.append({
                    "task": task,
                    "observation": None if self.lightweight_mode else (
                        observation.to(self.dataset_device)),
                    "action": None if self.lightweight_mode else action,
                    "reward": reward,
                    "return": total_reward,
                    "agent_trace": self.agent.get_trace(device=self.dataset_device) if (
                            self.collect_trace and not self.lightweight_mode) else None,
                    "done": done,
                    "md": md
                })
                self.agent.clear_trace()

                observation = next_observation

                if done:
                    self.agent.finalize(total_reward)
                    rollout.append({
                        "task": task,
                        "observation": None if self.lightweight_mode else next_observation.to(self.dataset_device),
                        "action": None,
                        "agent_trace": None,
                        "reward": 0,
                        "return": total_reward,
                        "done": True,
                        "md": md  # TODO: This gets added twice, which might be confusing
                    })
                    new_ret = None
                    break
                else:
                    action = self.agent.act(observation)

                if t - start > max_section:
                    new_ret = {
                        "t": t,
                        "task": task,
                        "rollout_idx": rollout_idx,
                        "observation": observation,
                        "action": action
                    }
                    break
                else:
                    new_ret = None

            if new_ret is not None:
                print(f"Pause rollout: {self.counter}, length: {len(rollout)}")
                return rollout, new_ret, False
            else:
                print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                self.counter += 1
                return rollout, new_ret, True

    def rollout(self, skip_tasks=None):
        rollout = []
        with torch.no_grad():
            observation, task, rollout_idx = self.env.reset(skip_tasks=skip_tasks)

            # Skipped:
            if task is None:
                return None

            print("Task: ", str(task))
            self.agent.start_new_rollout(task)

            action = self.agent.act(observation)
            total_reward = 0
            for t in range(self.horizon):
                #print(f"Taking action: {action}")
                next_observation, reward, done, md = self.env.step(action)
                total_reward += reward

                rollout.append({
                    "task": task,
                    "observation": None if self.lightweight_mode else (
                        observation.to(self.dataset_device)),
                    "action": None if self.lightweight_mode else action,
                    "reward": reward,
                    "return": total_reward,
                    "agent_trace": self.agent.get_trace(device=self.dataset_device) if (
                            self.collect_trace and not self.lightweight_mode) else None,
                    "done": done,
                    "md": md
                })
                self.agent.clear_trace()

                observation = next_observation

                if done:
                    self.agent.finalize(total_reward)
                    rollout.append({
                        "task": task,
                        "observation": None if self.lightweight_mode else next_observation.to(self.dataset_device),
                        "action": None,
                        "agent_trace": None,
                        "reward": 0,
                        "return": total_reward,
                        "done": True,
                        "md": md # TODO: This gets added twice, which might be confusing
                    })
                    break
                else:
                    action = self.agent.act(observation)

            print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
            self.counter += 1
            return rollout

    def rollout_and_send(self):
        self._load_agent_state_from_ps()
        rollout = self.rollout()

        # Send to the dataset process
        self.dataset_process.add_rollout.remote(rollout)

        # Write metrics to tensorboard
        #if self.writer is not None:
        #    metrics = get_multiple_rollout_metrics_bw([rollout])
        #    self.writer.add_scalar_dict("tsa_rollout", metrics)
        #    self.writer.inc_iter()
        return


@ray.remote(num_cpus=1, num_gpus=0)
class RolloutActor(RolloutActorLocal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)