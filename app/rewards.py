"""
app/rewards.py — Deterministic reward computation for the Study Planner Environment.

Bug fixes from uploaded code
─────────────────────────────
  • random.random() REMOVED — all rewards are deterministic; same input → same output.
  • grade() unified into compute_reward() and compute_final_score() with clear signatures.
  • All penalties are fixed constants, not probabilistic.

Per-step reward breakdown
──────────────────────────
  mark_complete (on time)  : +1.00
  mark_complete (late)     : +0.40  (partial credit — still finished)
  schedule_task            : +0.15  (planning bonus)
  take_break               : +0.10  (self-care is rewarded)
  skip_task                : -0.50  (abandonment penalty)
  burnout penalty          : -0.30  (energy < 20)
  stress overload penalty  : -0.20  (stress > 80, medium/hard only)
  missed deadline penalty  : -0.50 per newly missed deadline (applied once)
  reasoning bonus          : +0.10  (any non-empty reasoning string)

Final episode score (used by /grader and /baseline)
─────────────────────────────────────────────────────
  completion_score   = completed / total_tasks              × 0.5 (easy) / 0.4 (medium) / 0.35 (hard)
  deadline_score     = (total - missed) / total             × 0.3
  stress_score       = max(0, 1 − stress/100)              × 0.2 (easy) / 0.2 (med) / 0.2 (hard)
  energy_score       = max(0, energy/100)                  × 0.0 (easy) / 0.1 (med) / 0.15 (hard)
  missed_penalty     = missed_count × 0.20
  Final = max(0, min(1, weighted_sum - missed_penalty))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Weights by task difficulty
# ---------------------------------------------------------------------------

_WEIGHTS: Dict[str, Dict[str, float]] = {
    "study-schedule": {
        "completion": 0.50,
        "deadline":   0.30,
        "stress":     0.20,
        "energy":     0.00,
        "missed_penalty_per_task": 0.20,
        "stress_threshold": 100,   # no per-step stress penalty on easy
        "energy_threshold": 20,
        "stress_penalty_per_step": 0.0,
    },
    "study-manage": {
        "completion": 0.40,
        "deadline":   0.30,
        "stress":     0.20,
        "energy":     0.10,
        "missed_penalty_per_task": 0.20,
        "stress_threshold": 70,
        "energy_threshold": 20,
        "stress_penalty_per_step": 0.20,
    },
    "study-optimize": {
        "completion": 0.35,
        "deadline":   0.30,
        "stress":     0.20,
        "energy":     0.15,
        "missed_penalty_per_task": 0.25,
        "stress_threshold": 50,
        "energy_threshold": 30,
        "stress_penalty_per_step": 0.25,
    },
}


# ---------------------------------------------------------------------------
# Per-step reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    action_type: str,
    task_name: Optional[str],
    completed_on_time: bool,
    already_missed: bool,
    newly_missed_count: int,
    energy: int,
    stress: int,
    task: str,
    reasoning: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute per-step reward deterministically.

    Parameters
    ----------
    action_type       : the action submitted by the agent
    task_name         : task targeted (may be None for take_break)
    completed_on_time : True if mark_complete and current_time <= deadline
    already_missed    : True if task was already past deadline (late completion)
    newly_missed_count: how many deadlines just passed this step (for penalty)
    energy            : agent's energy AFTER the step
    stress            : agent's stress AFTER the step
    task              : active task id (study-schedule | study-manage | study-optimize)
    reasoning         : optional justification string

    Returns
    -------
    (reward: float, info: dict)
    """
    w = _WEIGHTS.get(task, _WEIGHTS["study-manage"])
    breakdown: Dict[str, float] = {
        "action_reward":       0.0,
        "burnout_penalty":     0.0,
        "stress_penalty":      0.0,
        "missed_penalty":      0.0,
        "reasoning_bonus":     0.0,
    }

    # ── Action reward ───────────────────────────────────────────────────
    if action_type == "mark_complete":
        if completed_on_time:
            breakdown["action_reward"] = 1.00
        else:
            breakdown["action_reward"] = 0.40   # late but still finished
    elif action_type == "schedule_task":
        breakdown["action_reward"] = 0.15
    elif action_type == "take_break":
        breakdown["action_reward"] = 0.10
    elif action_type == "skip_task":
        breakdown["action_reward"] = -0.50

    # ── Burnout penalty (deterministic threshold, no randomness) ────────
    if energy < w["energy_threshold"]:
        breakdown["burnout_penalty"] = -0.30

    # ── Stress overload penalty (medium/hard only) ───────────────────────
    if stress > w["stress_threshold"] and w["stress_penalty_per_step"] > 0:
        breakdown["stress_penalty"] = -w["stress_penalty_per_step"]

    # ── Missed deadline penalty (applied once when deadline first passes) ─
    if newly_missed_count > 0:
        breakdown["missed_penalty"] = -0.50 * newly_missed_count

    # ── Reasoning bonus ─────────────────────────────────────────────────
    if reasoning and len(reasoning.strip()) > 3:
        breakdown["reasoning_bonus"] = 0.10

    raw = sum(breakdown.values())
    reward = round(max(-1.0, min(1.0, raw)), 4)

    info: Dict[str, Any] = {
        "action_type":   action_type,
        "task_name":     task_name,
        "breakdown":     breakdown,
        "reward":        reward,
        "energy":        energy,
        "stress":        stress,
    }
    return reward, info


# ---------------------------------------------------------------------------
# Final episode score  (matches grader.py intent, fixed signature)
# ---------------------------------------------------------------------------

def compute_final_score(
    total_tasks: int,
    completed_count: int,
    missed_count: int,
    final_energy: int,
    final_stress: int,
    task: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the final episode score in [0, 1].

    Parameters
    ----------
    total_tasks     : total tasks in this episode
    completed_count : tasks successfully marked complete
    missed_count    : tasks whose deadline passed without completion
    final_energy    : energy level at episode end
    final_stress    : stress level at episode end
    task            : active task id

    Returns
    -------
    (score: float in [0,1], breakdown: dict)
    """
    w = _WEIGHTS.get(task, _WEIGHTS["study-manage"])

    n = max(total_tasks, 1)

    completion_score = completed_count / n
    deadline_score   = (n - missed_count) / n
    stress_score     = max(0.0, 1.0 - final_stress / 100)
    energy_score     = max(0.0, final_energy / 100)

    missed_penalty = missed_count * w["missed_penalty_per_task"]

    weighted = (
        w["completion"] * completion_score
        + w["deadline"]   * deadline_score
        + w["stress"]     * stress_score
        + w["energy"]     * energy_score
    )

    score = round(max(0.0, min(1.0, weighted - missed_penalty)), 4)

    breakdown: Dict[str, Any] = {
        "completion_score": round(completion_score, 4),
        "deadline_score":   round(deadline_score, 4),
        "stress_score":     round(stress_score, 4),
        "energy_score":     round(energy_score, 4),
        "missed_penalty":   round(-missed_penalty, 4),
        "weights":          {k: v for k, v in w.items() if isinstance(v, float)},
        "final_score":      score,
    }
    return score, breakdown