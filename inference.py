"""
inference.py -- HydroPulse AI Agent (Hackathon Submission)

Runs a full episode against the HydroPulse environment server using an LLM
to decide turbine and spillway release actions. Prints structured logs consumed by the judge.

Required environment variables:
    API_BASE_URL  -- OpenAI-compatible API base URL
    MODEL_NAME    -- Model identifier (e.g. "meta-llama/Llama-3.3-70B-Instruct")
    HF_TOKEN      -- Hugging Face / API key

Optional environment variables:
    ENV_URL           -- Environment server URL (defaults to deployed HF Space)
    LOCAL_IMAGE_NAME  -- If using from_docker_image()

Usage:
    python inference.py
    python inference.py --task baseline_generation
    ENV_URL=http://localhost:8000 python inference.py

STDOUT FORMAT (machine-parsed by judge -- do not change):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import argparse
import os
import sys
import json
from typing import List, Optional

from openai import OpenAI

# ── Import the environment client ──────────────────────────────────────────────
try:
    from HydroPulse.client import HydropulseEnv
    from HydroPulse.models import HydropulseAction
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from client import HydropulseEnv
    from models import HydropulseAction


# ── Constants ──────────────────────────────────────────────────────────────────
ENV_URL          = os.environ.get("ENV_URL", "https://kenzhok-hydropulse.hf.space")
API_BASE_URL     = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")  # active default
MODEL_NAME       = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")  # active default
HF_TOKEN         = os.environ.get("HF_TOKEN")   # NO default — must be set explicitly
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")  # optional — used with from_docker_image()

BENCHMARK = "HydroPulse"
VALID_TASKS = {"baseline_generation", "peak_shaving", "storm_surge"}


# ── Mandatory logging helpers (judge-parsed format) ───────────────────────────
def log_start(task: str, env_name: str, model: str) -> None:
    """[START] line — emitted exactly once at episode begin."""
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    """[STEP] line — emitted immediately after each env.step() returns."""
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """[END] line — always emitted (even on exception) via finally block."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM setup ─────────────────────────────────────────────────────────────────
def get_llm_client() -> OpenAI:
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set.\n"
            "Example: set HF_TOKEN=hf_your_token_here"
        )
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(obs) -> str:
    return f"""You are a Hydro-Plant Management AI controlling a dam with two release valves.

Your goal is to MAXIMISE revenue while PREVENTING reservoir overflow and downstream flooding.

Current State:
- reservoir_level: {obs.reservoir_level:.2f} (safe zone: 40.0 to 60.0, max capacity: 100.0)
- inflow_rate: {obs.inflow_rate:.2f} (water flowing in per step)
- grid_demand_price: {obs.grid_demand_price:.2f} (higher = more revenue per unit turbine release)
- downstream_capacity: {obs.downstream_capacity:.2f} (max total release before flooding)

Rules:
- turbine_release (0.0-1.0) generates revenue = turbine_release * grid_demand_price
- spillway_release (0.0-1.0) dumps water but earns no revenue
- total_release = (turbine_release * 10.0) + (spillway_release * 50.0)
- If total_release > downstream_capacity -> flood penalty (reward = 0.0)
- If reservoir_level > 100.0 -> dam breach penalty (reward = 0.0)
- Keep reservoir between 40-60 for a bonus

Respond ONLY in this JSON format:
{{"turbine_release": <float 0.0-1.0>, "spillway_release": <float 0.0-1.0>}}"""


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(client: OpenAI, obs) -> HydropulseAction:
    """Call the LLM and parse turbine/spillway releases. Falls back to heuristic on error."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(obs)}],
            max_tokens=64,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return HydropulseAction(
            turbine_release=float(data.get("turbine_release", 0.5)),
            spillway_release=float(data.get("spillway_release", 0.0)),
        )
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", file=sys.stderr, flush=True)
        return _heuristic_action(obs)


def _heuristic_action(obs) -> HydropulseAction:
    """Safe fallback: open spillway if high, otherwise generate power."""
    if obs.reservoir_level > 80.0:
        return HydropulseAction(turbine_release=0.5, spillway_release=0.8)
    return HydropulseAction(turbine_release=0.5, spillway_release=0.0)


# ── Score computation ─────────────────────────────────────────────────────────
def compute_score(rewards: List[float]) -> float:
    """Return mean per-step reward, clamped strictly to (0.01, 0.99)."""
    if not rewards:
        return 0.50
    mean_reward = sum(rewards) / len(rewards)
    return min(max(round(mean_reward, 4), 0.01), 0.99)


# ── Main episode loop ─────────────────────────────────────────────────────────
def run_episode(task: str) -> None:
    llm_client  = get_llm_client()
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    with HydropulseEnv(base_url=ENV_URL) as env:
        log_start(task=task, env_name=BENCHMARK, model=MODEL_NAME)

        try:
            result = env.reset()
            obs    = result.observation

            while not obs.done:
                steps_taken += 1
                action = call_llm(llm_client, obs)

                error: Optional[str] = None
                try:
                    step_result = env.step(action)
                    obs         = step_result.observation
                    reward      = step_result.reward or 0.0
                except Exception as e:
                    reward = 0.0
                    error  = str(e)
                    rewards.append(reward)
                    log_step(step=steps_taken, action=str(action.model_dump()), reward=reward, done=True, error=error)
                    break

                rewards.append(reward)
                log_step(
                    step=steps_taken,
                    action=str(action.model_dump()),
                    reward=reward,
                    done=obs.done,
                    error=error,
                )

            score   = compute_score(rewards)
            success = score > 0.333

        except Exception as outer_exc:
            print(f"[DEBUG] Episode error: {outer_exc}", file=sys.stderr, flush=True)

        finally:
            # [END] MUST always be emitted — even on exception
            log_end(
                success=success,
                steps=steps_taken,
                score=score,
                rewards=rewards,
            )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HydroPulse AI Agent -- Hackathon inference script"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["baseline_generation", "peak_shaving", "storm_surge", "all"],
        help="Task to run (default: all)",
    )
    args = parser.parse_args()

    # Allow judge to override via TASK_NAME env var
    target_task = os.getenv("TASK_NAME")
    if target_task:
        if "baseline" in target_task:   args.task = "baseline_generation"
        elif "peak" in target_task:     args.task = "peak_shaving"
        elif "storm" in target_task:    args.task = "storm_surge"

    tasks = list(VALID_TASKS) if args.task == "all" else [args.task]

    for t in tasks:
        run_episode(task=t)
