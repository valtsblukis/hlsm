from typing import Dict
from lgp.abcd.factory import Factory

from lgp.env.alfred.alfred_env import AlfredEnv
from lgp.agents.agents import get_agent
from lgp.models.alfred.hlsm.hlsm_model_factory import HlsmModelFactory

from lgp.parameters import Hyperparams


class AlfredFactory(Factory):
    def __init__(self):
        super().__init__()

    def get_model_factory(self, setup: Dict, hparams : Hyperparams):
        # TODO support picking between multiple models if needed
        return HlsmModelFactory(hparams)

    def get_environment(self, setup : Dict, task_num_range=None):
        # TODO: Retrieve train/dev/test split based on setup
        device = setup.get("device", "cpu")
        env = AlfredEnv(device=device, setup=setup["env_setup"])
        env.set_task_num_range(task_num_range)
        return env

    def get_agent(self, setup : Hyperparams, hparams : Hyperparams):
        return get_agent(setup, hparams)

