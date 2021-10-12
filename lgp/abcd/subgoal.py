from abc import ABC, abstractmethod


class Subgoal(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def type_id(self):
        ...

    @abstractmethod
    def is_stop(self) -> bool:
        """
        :return: True if this is a STOP-action, False otherwise.
        """
        ...