from typing import Union
from enum import Enum

from lgp.abcd.action import Action

from lgp.env.blockworld.state.direction import Direction


class ActionArgument():
    ...

class ActionType(Enum):
    NAV = "NAV"
    PICKUP = "PICKUP"
    STOP = "STOP"
    DROP = "DROP"

    def __str__(self):
        return str(self.name)


class NavigateArgument(ActionArgument):
    def __init__(self, direction: Direction):
        self.direction = direction

    def __eq__(self, other: "NavigateArgument"):
        return self.direction == other.direction

    def __str__(self):
        return str(f"->{self.direction}")

class PickupArgument(ActionArgument):
    def __init__(self, coord: (int, int)):
        self.coord = coord

    def __eq__(self, other: "PickupArgument"):
        return self.coord[0] == other.coord[0] and self.coord[1] == other.coord[1]

    def __str__(self):
        return f"^({self.coord[0]},{self.coord[1]})"


class BwAction(Action):
    def __init__(self, type: ActionType, argument: Union[ActionArgument, None]):
        super().__init__()
        self.type = type
        self.argument = argument

    def is_stop(self):
        return self.type == ActionType.STOP

    def __eq__(self, other: "BwAction"):
        return self.type == other.type and self.argument == other.argument

    def __str__(self):
        return f"{self.type} : {self.argument}"
        #return f"Blockworld Action of type={self.type}, argument={self.argument}"