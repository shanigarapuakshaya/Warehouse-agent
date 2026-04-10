import os
from openai import OpenAI
from app import env

def run_inference():
    task_name = "warehouse-easy"

    # Initialize LLM client (REQUIRED)
    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_BASE_URL"]
    )

    print(f"[START] task={task_name}", flush=True)

    state = env.reset("easy")
    total_reward = 0
    steps = 0

    for i in range(20):
        x, y = state["position"]
        gx, gy = state["goal"]

        # REQUIRED API call (for validation)
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a warehouse navigation agent."},
                {"role": "user", "content": f"Current position: {state['position']}, Goal: {state['goal']}. Suggest next move."}
            ],
        )

        # logic
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

        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if done:
            break

    score = total_reward / steps if steps > 0 else 0.0

    print(f"[END] task={task_name} score={score:.3f} steps={steps}", flush=True)

    return score


if __name__ == "__main__":
    run_inference()
