import os
import json
from openai import OpenAI
from client import HydropulseEnv
from models import HydropulseAction

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")

    # Only instantiate OpenAI if we have an API key
    if api_key:
        llm = OpenAI(api_key=api_key, base_url=api_base_url)
    else:
        llm = None
        print("Warning: OPENAI_API_KEY not found. Will use heuristic fallback.")

    print("[START]")

    total_reward = 0.0

    with HydropulseEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        done = result.done

        while not done:
            obs = result.observation
            
            action = None
            if llm is not None:
                try:
                    # Instruct the model to return JSON with turbine_release and spillway_release
                    response = llm.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a Hydro-Plant Management AI. Your goal is to maximize revenue while avoiding floods and breaches. Return raw JSON with 'turbine_release' and 'spillway_release' float values between 0.0 and 1.0."},
                            {"role": "user", "content": f"Observation: {obs.model_dump_json()}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    action_data = json.loads(response.choices[0].message.content)
                    action = HydropulseAction(
                        turbine_release=float(action_data.get("turbine_release", 0.5)),
                        spillway_release=float(action_data.get("spillway_release", 0.0))
                    )
                except Exception as e:
                    print(f"LLM Error, falling back to heuristic: {e}")
            
            # Heuristic action if LLM fails or is missing
            if action is None:
                # Basic heuristic: if level > 80, open spillway. Otherwise generate base power.
                if obs.reservoir_level > 80.0:
                    action = HydropulseAction(turbine_release=1.0, spillway_release=1.0)
                else:
                    action = HydropulseAction(turbine_release=0.5, spillway_release=0.0)

            print(f"[STEP] {action.model_dump()}")
            
            result = env.step(action)
            total_reward += result.reward
            done = result.done

    print("[END]")

if __name__ == "__main__":
    main()
