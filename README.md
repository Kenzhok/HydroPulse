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

# 💧 HydroPulse — Hydroelectric Dam Management Environment

> **Meta × Scaler OpenEnv AI Hackathon Submission**

HydroPulse is an OpenEnv-compatible benchmark for safety-constrained decision-making in physical infrastructure, providing a compact, reusable testbed for continuous control, temporal planning, and safety-critical resource allocation under uncertainty. It fills a crucial gap in OpenEnv's ecosystem—which is dominated by language-heavy workflow environments—by driving agents against a stochastic, non-linear physical fluid mechanics engine. Inspired by Japan's G-Cans flood-defense system, the AI evaluates risk tradeoffs by acting as an automated downstream operator, balancing proactive disaster prevention against peak-revenue energy generation.

---

## 🌊 Environment Overview

At each step the agent chooses how much water to release through two valves:

| Valve | Control | Effect |
|---|---|---|
| **Turbine** | `turbine_release` (0.0–1.0) | Generates electricity revenue, max 10 units/step |
| **Spillway** | `spillway_release` (0.0–1.0) | Dumps excess water, max 30 units/step |

**Core physics (Torricelli hydraulic head & evaporation):**
```python
head_pressure = math.sqrt(max(0, current_level) / 100.0)
evap_loss = 0.05 * (current_level ** 0.66)
new_level = current_level + inflow_rate
          - (turbine_release * 10.0 * head_pressure)
          - (spillway_release * 30.0 * head_pressure)
          - evap_loss
```

## 🎯 Who Is This For?
* **RL researchers** who need a physically grounded, reusable benchmark with meaningful failure states
* **Agent evaluation teams** testing whether frontier LLMs can reason about continuous quantitative control under delayed consequences
* **Post-training researchers** looking for environments that require multi-step physical planning rather than single-turn text generation

---

## 🎮 Action Space

```python
class HydropulseAction(Action):
    turbine_release:  float  # 0.0 to 1.0
    spillway_release: float  # 0.0 to 1.0
```

## 👁️ Observation Space

```python
class HydropulseObservation(Observation):
    reservoir_level:     float  # Current water level (clamped to 0–100)
    inflow_rate:         float  # Stochastic water entering reservoir per step
    grid_demand_price:   float  # Revenue multiplier (diurnal sine wave + noise)
    downstream_capacity: float  # Max safe total outflow = 40.0
    value:               float  # Reward earned this step
```

---

## 🏆 Reward Function

All step rewards are strictly normalised to **[0.0, 1.0]**:

| Condition | Reward |
|---|---|
| Reservoir overflows (`level > 100`) | **0.0** — dam breach |
| Reservoir depleted (`level < 0`) | **0.0** — physically impossible |
| Downstream flood (`total_release > 40.0`) | **0.0** — flood constraint violated |
| Clean operation | `(actual_turbine_flow × price) / 800.0` |

> Max possible revenue = `MAX_TURBINE_FLOW × MAX_PRICE` (`10.0 × 80.0 = 800.0`). The reward strictly normalizes revenue against this theoretical maximum. Because reward scales with both turbine flow and current price, the environment creates a genuine tension between holding water for future price peaks (medium) and releasing water before a flood (hard) — these objectives are mutually exclusive at the same timestep.

---

## 📋 Tasks & Graders

### 🟢 Easy — Baseline Generation
Steady inflow of **5 units/step**. No surprises.

**Challenge:** Keep the reservoir level between 40% and 60% while generating power.

**Grader:** `% of steps where reservoir_level ∈ [40.0, 60.0]`

**Optimal strategy:** `turbine=0.5, spillway=0.0` → net flow = 0, level stays at 50.

---

### 🟡 Medium — Peak Shaving
Steady inflow of 5 units/step. `grid_demand_price` follows a **diurnal sine wave** combined with stochastic Gaussian noise, simulating realistic energy markets.

**Challenge:** Maximise diurnal revenue by timing turbine output to evening peaks while managing changing hydraulic head pressures.

**Grader:** `total revenue earned / maximum theoretically possible revenue`

**Optimal strategy:** Ramp up turbine release as prices rise.

---

### 🔴 Hard — Storm Surge
At **step 5**, inflow surges from `5.0 → 42.0` for 10 steps (steps 5–14) (simulating a monsoon), then returns to `5.0`.

**Challenge:** With surge inflow of 42 units/step exceeding the maximum reactive release capacity (~28 units at 50% level), the agent must proactively drain the reservoir below 15% before step 5 to survive the storm surge.

**Grader:** `1.0` if **zero** breaches throughout the episode, `0.0` if any constraint is ever violated.

**Constraints:** `total_release ≤ 40.0` per step (spillway alone can handle surge if turbine is also running).

**Optimal pre-surge strategy:** Run turbine fully to drop level below 15 before step 5, then open spillway during surge.

---

## 🧠 Why Is This Hard?

Torricelli head pressure makes the environment non-linear — releasing water from a 90% full dam is not 3× more effective than from a 30% full dam; it is only ~√3 ≈ 1.73× more effective. Agents that assume linearity will systematically over-release from low levels and under-release from high levels.

The Hard tier forces proactive planning because at 50% reservoir level the maximum reactive release (~28 units/step) is lower than the surge inflow (42 units/step), making overflow mathematically certain unless the agent pre-drains below ~15% before step 5.

Gaussian price noise (σ=5) and stochastic inflow (±1.5) mean the optimal action at any step cannot be computed from a lookup table — the agent must integrate uncertain signals across a 20-step horizon where a single overflow terminates the episode with zero reward.

---

## 🚀 Quick Start

```python
from client import HydropulseEnv
from models import HydropulseAction

with HydropulseEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    obs = result.observation

    while not obs.done:
        # Smart heuristic: open spillway when level is high
        if obs.reservoir_level > 75.0:
            action = HydropulseAction(turbine_release=0.5, spillway_release=0.8)
        else:
            action = HydropulseAction(turbine_release=0.5, spillway_release=0.0)

        result = env.step(action)
        obs = result.observation
        print(f"Level: {obs.reservoir_level:.1f} | Reward: {result.reward:.3f}")
```

---

## 🏃 Running Locally

```bash
# 1. Install dependencies
uv sync

# 2. Start the server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# 3. Validate the environment
uv run openenv validate

# 4. Run the AI agent
HF_TOKEN=hf_your_token python inference.py
```

---

## 🤖 Running Inference

The `inference.py` script runs the LLM agent through all three tasks and emits judge-compatible logs.

```bash
# Required
export HF_TOKEN=hf_your_token_here

# Optional overrides
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export ENV_URL=https://kenzhok-hydropulse.hf.space

python inference.py
```

**Output format (machine-parsed by judge):**
```
[START] task=baseline_generation env=HydroPulse model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={'turbine_release': 0.5, 'spillway_release': 0.0} reward=0.20 done=false error=null
...
[END] task=baseline_generation success=true steps=20 score=0.200 rewards=0.20,0.20,...
```

---

## 📦 Project Structure

```
HydroPulse/
├── Dockerfile                        # Container image definition
├── README.md                         # This document
├── openenv.yaml                      # OpenEnv manifest — tasks & metadata
├── pyproject.toml                    # Python build configuration
├── uv.lock                           # UV dependency lockfile
├── inference.py                      # LLM agent evaluation script
├── client.py                         # Environment WebSocket client payload
├── models.py                         # Pydantic schemas for observation/actions
└── server/
    ├── app.py                        # FastAPI server (HTTP/WS endpoints)
    ├── HydroPulse_environment.py     # Core physics engine & step physics
    ├── tasks.py                      # Difficulty grader math functions
    └── requirements.txt              # Server runtime dependencies
```

---

## 🔧 Environment Constants

| Constant | Value | Description |
|---|---|---|
| `MAX_CAPACITY` | `100.0` | Reservoir overflow threshold |
| `MAX_TURBINE_CAPACITY` | `10.0` | Max turbine flow (at release=1.0) |
| `MAX_SPILLWAY_CAPACITY` | `30.0` | Max spillway flow (at release=1.0) |
| `DOWNSTREAM_CAPACITY` | `40.0` | Max safe total outflow per step |
| Episode length | `20 steps` | Fixed per episode |

> **Design note:** `MAX_TURBINE + MAX_SPILLWAY = DOWNSTREAM_CAPACITY (40.0)` — this means at full spillway + full turbine the agent exactly reaches (but does not exceed) the flood threshold, making the Hard task genuinely solvable without being trivial.

---

## 📊 Benchmark Results (Heuristic Baseline)

| Task | Strategy | Avg Reward | Breaches |
|---|---|---|---|
| Easy | `turbine=0.5` steady | 0.200 | 0/20 |
| Medium | `turbine=0.5` + spike response | 0.290 | 0/20 |
| Hard | `spillway=0.8` pre-surge | 0.090 | ~5/20 |

---

*Built for the **Meta × Scaler OpenEnv AI Hackathon** by [@Kenzhok](https://huggingface.co/Kenzhok)*
