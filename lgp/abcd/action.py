from abc import ABC, abstractmethod, abstractclassmethod


class Action(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def is_stop(self) -> bool:
        """
        :return: True if this is a STOP-action, False otherwise.
        """
        ...
