import gradio as gr
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

# STATE
@dataclass
class State:
    agent_pos: Tuple[int, int]
    pickup_points: List[Tuple[int, int]]
    drop_zone: Tuple[int, int]
    carrying: bool

# ENVIRONMENT
class WarehouseEnv:
    def __init__(self, size=5):
        self.size = size

    def reset(self, mode="easy"):
        self.agent_pos = (0, 0)
        self.carrying = False

        if mode == "easy":
            self.pickup_points = [(2, 2)]
            self.obstacles = [(1, 1)]
        elif mode == "medium":
            self.pickup_points = [(2, 2), (3, 1)]
            self.obstacles = [(1,1), (2,3)]
        else:
            self.pickup_points = [(2,2), (3,1), (1,3)]
            self.obstacles = [(1,1), (2,3), (3,3)]

        self.drop_zone = (4, 4)
        self.total_tasks = len(self.pickup_points)
        return self.state()

    def state(self):
        return State(self.agent_pos, self.pickup_points.copy(), self.drop_zone, self.carrying)

    def step(self, action):
        x, y = self.agent_pos
        reward = -1
        done = False

        moves = {"up":(-1,0), "down":(1,0), "left":(0,-1), "right":(0,1)}

        if action in moves:
            dx, dy = moves[action]
            nx, ny = x+dx, y+dy

            if 0 <= nx < self.size and 0 <= ny < self.size:
                if (nx, ny) not in self.obstacles:
                    self.agent_pos = (nx, ny)
                else:
                    reward = -10

        elif action == "pickup":
            if self.agent_pos in self.pickup_points and not self.carrying:
                self.pickup_points.remove(self.agent_pos)
                self.carrying = True
                reward = 10
            else:
                reward = -10

        elif action == "drop":
            if self.agent_pos == self.drop_zone and self.carrying:
                self.carrying = False
                reward = 15
                if len(self.pickup_points) == 0:
                    done = True
            else:
                reward = -10

        return self.state(), reward, done, {}

# BFS
def get_next_step(start, goal, obstacles, size):
    queue = deque([(start, [])])
    visited = {start}
    moves = {"up":(-1,0), "down":(1,0), "left":(0,-1), "right":(0,1)}

    while queue:
        (x,y), path = queue.popleft()
        if (x,y) == goal:
            return path[0] if path else None

        for action,(dx,dy) in moves.items():
            nx, ny = x+dx, y+dy
            if 0 <= nx < size and 0 <= ny < size:
                if (nx,ny) not in obstacles and (nx,ny) not in visited:
                    queue.append(((nx,ny), path+[action]))
                    visited.add((nx,ny))
    return None

# AGENT
def agent(state, env):
    if len(state.pickup_points) == 0 and not state.carrying:
        return None

    x, y = state.agent_pos

    if not state.carrying:
        target = min(state.pickup_points, key=lambda p: abs(p[0]-x)+abs(p[1]-y))
    else:
        target = state.drop_zone

    if state.agent_pos == target:
        return "pickup" if not state.carrying else "drop"

    return get_next_step(state.agent_pos, target, env.obstacles, env.size)

# RUN
def run_env(mode):
    env = WarehouseEnv()
    state = env.reset(mode)
    output = ""

    for step in range(50):
        action = agent(state, env)
        if action is None:
            break
        state, reward, done, _ = env.step(action)
        output += f"Step {step}: {action}, Pos={state.agent_pos}, Reward={reward}\n"
"
        if done:
            break

    score = (env.total_tasks - len(env.pickup_points)) / env.total_tasks
    return output + f"\nFinal Score: {score:.3f}"

# UI
demo = gr.Interface(
    fn=run_env,
    inputs=gr.Dropdown(["easy","medium","hard"]),
    outputs="text",
    title="Warehouse Agent Simulation"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
