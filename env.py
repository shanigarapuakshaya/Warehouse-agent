from models import Observation, Reward
from tasks import TASKS, manhattan


class WarehouseEnv:
    def __init__(self):
        self.reset("easy")

    def reset(self, mode="easy"):
        self.mode = mode
        task = TASKS[mode]

        self.size = task["size"]
        self.position = task["start"]
        self.goal = task["goal"]
        self.item_picked = False
        self.steps = 0

        return self.state()

    def state(self):
        return Observation(
            position=self.position,
            goal=self.goal,
            item_picked=self.item_picked,
            steps=self.steps
        )

    def step(self, action):
        x, y = self.position
        self.steps += 1

        # movement
        if action.move == "up":
            x -= 1
        elif action.move == "down":
            x += 1
        elif action.move == "left":
            y -= 1
        elif action.move == "right":
            y += 1

        # boundary check
        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))

        self.position = [x, y]

        reward = 0.0
        done = False

        # reward shaping (important)
        dist = manhattan(self.position, self.goal)
        reward += (self.size * 2 - dist) / (self.size * 2)

        # pick logic
        if action.move == "pick" and self.position == self.goal:
            self.item_picked = True
            reward = 1.0
            done = True

        # failure condition
        if self.steps >= self.size * 3:
            done = True

        return self.state(), Reward(score=reward, reason="step update"), done, {}
