import sys
import inspect


AUTHORIZED_CALLERS = [
    # Allow oracles and privileged agent access
    "lgp.agents.blockworld.oracle_pickup_agent",
    "lgp.agents.blockworld.oracle_move_agent",
    "lgp.agents.blockworld.oracle_agent",
    "lgp.models.blockworld.privileged_model",

    # Allow observation class to expose certain properties of this explicitly
    "lgp.env.alfred.alfred_observation"]

class UnauthorizedAccessException(Exception):
    ...

class PrivilegedInfo:

    def __init__(self, world_state):
        self._world_state = world_state
        self._task = None

    def attach_task(self, task):
        self._task = task

    def __getattr__(self, item):
        caller_frame = sys._getframe(1)
        caller_name = inspect.getmodule(caller_frame).__name__
        # Only check calls from our own codebase. Otherwise we stop e.g. serialization, monitoring etc.
        if caller_name.startswith("lgp.") and caller_name not in AUTHORIZED_CALLERS:
            raise UnauthorizedAccessException(f"Caller {caller_name} not authorized to access PrivilegedInfo")
        if item == "world_state":
            return self._world_state

    # Whenever __getattr__ is used, these two must also be present for pickling to work
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)