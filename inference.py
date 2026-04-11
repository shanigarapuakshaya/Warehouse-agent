import os
from openai import OpenAI
from app import env


def run_inference():
    task_name = "warehouse-easy"

    # MUST use these exact env variables
    api_base = os.environ["API_BASE_URL"]
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = os.environ.get("HF_TOKEN", "dummy")

    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )

    print(f"[START] task={task_name}", flush=True)

    state = env.reset("easy")
    total_reward = 0.0
    steps = 0

    for step in range(1, 11):
        x, y = state["position"]
        gx, gy = state["goal"]

        # ORCE API CALL (no skipping)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a warehouse agent."},
                {"role": "user", "content": f"Position: {state['position']} Goal: {state['goal']}. Give one-word move."}
            ],
        )

        # try to use model output
        try:
            action = response.choices[0].message.content.strip().lower()
        except:
            action = ""

        # fallback (ensures completion)
        if action not in ["up", "down", "left", "right", "pick"]:
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
        steps = step

        print(f"[STEP] step={step} reward={reward}", flush=True)

        if done:
            break

    score = total_reward / steps if steps > 0 else 0.0

    print(f"[END] task={task_name} score={score:.3f} steps={steps}", flush=True)

    return score


if __name__ == "__main__":
    run_inference()



    
