from app import env

def run_inference():
    state = env.reset("easy")
    total_reward = 0

    for _ in range(50):
        x, y = state["position"]
        gx, gy = state["goal"]

        if x < gx:
            action = "down"
        elif x > gx:
            action = "up"
        elif y < gy:
            action = "right"
        elif y > gy:
            action = "left"
        else:
            action = "pick"

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return {
        "final_state": state,
        "total_reward": total_reward
    }


if __name__ == "__main__":
    print(run_inference())
