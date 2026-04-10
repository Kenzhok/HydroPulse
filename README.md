---
title: HydroPulse Environment
emoji: 💧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - flood-management
  - hydroelectric
---

# HydroPulse Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent manages a hydroelectric dam — balancing **revenue generation** against **flood prevention** in a continuous-state physics simulation inspired by Japan's G-Cans flood defence system.

Built for the **Meta × Scaler OpenEnv AI Hackathon**.

---

## Environment Overview

The agent controls two release valves on a reservoir each step:
- **`turbine_release`** (0.0–1.0): Generates electricity revenue but has limited flow capacity.
- **`spillway_release`** (0.0–1.0): Dumps excess water downstream without generating revenue.

The physics engine updates the reservoir level each step:
```
new_level = current_level + inflow_rate
          - (turbine_release × 10.0)
          - (spillway_release × 50.0)
```

---

## Action Space

```python
class HydropulseAction(Action):
    turbine_release: float   # 0.0 to 1.0 — controls turbine flow
    spillway_release: float  # 0.0 to 1.0 — controls spillway flow
```

## Observation Space

```python
class HydropulseObservation(Observation):
    reservoir_level: float      # Current water level (0–100)
    inflow_rate: float          # Water flowing in per step
    grid_demand_price: float    # Revenue multiplier for turbine output
    downstream_capacity: float  # Max safe total release (40.0)
    value: float                # Reward earned on this step
```

---

## Reward Function

All rewards are strictly normalised to **[0.0, 1.0]**:

| Situation | Reward |
|---|---|
| Reservoir breach (`level > 100`) | **0.0** |
| Downstream flood (`total_release > 40.0`) | **0.0** |
| Normal operation | `(turbine_release × grid_demand_price) / 5.0` |
| Level within safe buffer (40–60%) | `+0.1` bonus |

---

## Tasks & Graders

### 🟢 Easy — Baseline Generation
Steady inflow. The agent must keep the water level between **40% and 60%**.
- **Grader**: % of steps where reservoir stayed in the safe buffer zone.

### 🟡 Medium — Peak Shaving
Low inflow with a `grid_demand_price` spike at step 10.
- **Grader**: Revenue generated / Maximum theoretically possible revenue.

### 🔴 Hard — Storm Surge
Massive inflow spike at step 5 simulating a monsoon.
- **Grader**: `1.0` if no breach or flood occurred, `0.0` if any constraint was violated.

---

## Quick Start

```python
from HydroPulse.client import HydropulseEnv
from HydroPulse.models import HydropulseAction

with HydropulseEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    obs = result.observation

    while not obs.done:
        # Choose releases based on current state
        if obs.reservoir_level > 80.0:
            action = HydropulseAction(turbine_release=0.5, spillway_release=0.8)
        else:
            action = HydropulseAction(turbine_release=0.5, spillway_release=0.0)

        result = env.step(action)
        obs = result.observation
        print(f"Level: {obs.reservoir_level:.1f} | Reward: {result.reward:.2f}")
```

---

## Running Locally

```bash
# Install dependencies
uv sync

# Start the server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# Validate environment
uv run openenv validate

# Run inference
HF_TOKEN=hf_your_token python inference.py
```

---

## Running Inference

```bash
# Required
export HF_TOKEN=hf_your_token_here

# Optional overrides
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export ENV_URL=https://kenzhok-hydropulse.hf.space

python inference.py
```

---

## Project Structure

```
HydroPulse/
├── README.md                          # This file
├── openenv.yaml                       # OpenEnv manifest (tasks + metadata)
├── pyproject.toml                     # Project dependencies
├── inference.py                       # AI agent inference script
├── client.py                          # HydropulseEnv client
├── models.py                          # Action & Observation models
└── server/
    ├── app.py                         # FastAPI server (HTTP + WebSocket)
    ├── HydroPulse_environment.py      # Core physics engine & reward shaping
    ├── tasks.py                       # Easy / Medium / Hard task graders
    └── Dockerfile                     # Container image
```
