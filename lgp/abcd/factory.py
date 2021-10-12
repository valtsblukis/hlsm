from typing import Dict
from abc import abstractmethod

from lgp.abcd.agent import Agent
from lgp.abcd.model_factory import ModelFactory
from lgp.abcd.env import Env

from lgp.parameters import Hyperparams


class Factory:

    def __init__(self):
        ...

    @abstractmethod
    def get_model_factory(self, setup: Dict, hparams : Hyperparams) -> ModelFactory:
        ...

    @abstractmethod
    def get_environment(self, setup: Dict) -> Env:
        ...

    @abstractmethod
    def get_agent(self, setup: Hyperparams, hparams: Hyperparams) -> Agent:
        ...
