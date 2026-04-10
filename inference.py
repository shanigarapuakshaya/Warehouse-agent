from app import env

def run_inference():
    task_name = "warehouse-easy"

    # START block
    print(f"[START] task={task_name}", flush=True)

    state = env.reset("easy")
    total_reward = 0
    steps = 0

    for i in range(50):
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
        steps += 1

        # STEP block
        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if done:
            break

    score = total_reward / steps if steps > 0 else 0.0

    # END block
    print(f"[END] task={task_name} score={score:.3f} steps={steps}", flush=True)

    return score


if __name__ == "__main__":
    run_inference()
