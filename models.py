from pydantic import BaseModel
from typing import Literal

class Observation(BaseModel):
    position: list[int]
    goal: list[int]
    item_picked: bool
    steps: int


class Action(BaseModel):
    move: Literal["up", "down", "left", "right", "pick"]


class Reward(BaseModel):
    score: float
    reason: str
