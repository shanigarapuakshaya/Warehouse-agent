from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Action(BaseModel):
    action: str

class WarehouseEnv:
    def __init__(self):
        self.reset("easy")

    def reset(self, mode="easy"):
        if mode == "easy":
            self.size = 4
        elif mode == "medium":
            self.size = 6
        else:
            self.size = 8

        self.position = [0, 0]
        self.goal = [self.size - 1, self.size - 1]
        self.item_picked = False
        self.steps = 0

        return self.state()

    def state(self):
        return {
            "position": self.position,
            "goal": self.goal,
            "item_picked": self.item_picked,
            "steps": self.steps
        }

    def step(self, action):
        x, y = self.position
        self.steps += 1

        if action == "up":
            x -= 1
        elif action == "down":
            x += 1
        elif action == "left":
            y -= 1
        elif action == "right":
            y += 1

        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))

        self.position = [x, y]

        reward = 0.0
        done = False

        dist = abs(x - self.goal[0]) + abs(y - self.goal[1])
        reward += (self.size * 2 - dist) / (self.size * 2)

        if action == "pick" and self.position == self.goal:
            self.item_picked = True
            reward = 1.0
            done = True

        if self.steps >= self.size * 3:
            done = True

        return self.state(), reward, done, {}

env = WarehouseEnv()

@app.get("/")
def home():
    return {"status": "Warehouse OpenEnv Running"}

@app.post("/reset")
def reset(mode: str = "easy"):
    return env.reset(mode)

@app.post("/step")
def step(action: Action):
    state, reward, done, _ = env.step(action.action)
    return {
        "state": state,
        "reward": reward,
        "done": done
    }

@app.get("/state")
def state():
    return env.state()


def main():
    return app


if __name__ == "__main__":
    main()
