# HydroPulse Environment Context

This file serves as a summary of the current state of the **HydroPulse-v1** project for handoff to a new conversation or AI context window.

## Overview
- **Project**: `HydroPulse-v1`
- **Framework**: Meta PyTorch OpenEnv Hackathon (Server & Client scaffold)
- **Goal**: A continuous-state Hydro-Plant Management environment designed for renewable energy generation and flood defense.

## Environment Definitions
The core physics map the water levels across steps:
`new_level = current_level + inflow_rate - (turbine_release * 10.0) - (spillway_release * 50.0)`

### Models (`models.py`)
- **Action**: `turbine_release` (0.0 to 1.0) and `spillway_release` (0.0 to 1.0).
- **Observation**: Monitors `reservoir_level`, `inflow_rate`, `grid_demand_price`, `downstream_capacity`, and a custom `value` field tracking the reward calculated at that exact step.

### Tasks & Graders (`server/tasks.py`)
1. **Easy (Baseline Generation)**: Checks what percentage of steps the agent maintained the water level exactly inside the safe operational buffer (40% to 60%). 
2. **Medium (Peak Shaving)**: Simulates a grid demand price spike. The grader tracks the revenue harvested vs the theoretical maximum revenue possible.
3. **Hard (Storm Surge)**: A massive monsoon surge forces rapid action. The grader gives `1.0` if the valley downstream wasn't flooded and the reservoir capacity didn't breach, otherwise `0.0`.

## Architecture Details
- **`server/HydroPulse_environment.py`**: Calculates the continuous level tracking, constraints, and dynamic scenario variables. Emits partial penalties for breaching capacity limitations.
- **`inference.py`**: A generic script hooked directly up to the `openai` Python SDK. It generates a single JSON sequence for the AI, tracks the required `[START]`, `[STEP]`, and `[END]` tags to satisfy the Meta PyTorch OpenEnv validator loop.
- **`openenv.yaml`**: The basic OpenEnv metadata schema registering the app interface and 3 tasks.

## Missing / Pending Components vs Standard Environments
The `HydroPulse` environment passed `uv run openenv validate` but currently does NOT have explicitly configured definitions for `action_space` and `observation_space` nested sequentially inside the `openenv.yaml`. This may optionally need to be implemented for nice Hugging Face UI layouts! 

## Next Action Required
- Run `uv run openenv push` to deploy it out to the Hugging Face space repository for evaluation.
