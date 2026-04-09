"""
app/rewards.py — Deterministic reward computation for the Email Triage Environment.
FIXED: Ensures all scores are strictly between 0 and 1 (never exactly 0 or 1)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSILON = 1e-6  # Small value to keep away from boundaries
MIN_REWARD = EPSILON
MAX_REWARD = 1.0 - EPSILON

_PRIORITY_RANK: Dict[str, int] = {
    "low": 0, "medium": 1, "high": 2, "critical": 3,
}

_GREETING = re.compile(
    r"\b(dear|hello|hi|greetings|good morning|good afternoon)\b", re.IGNORECASE
)
_CLOSING = re.compile(
    r"\b(sincerely|regards|best regards|kind regards|thank you|yours truly|warmly)\b",
    re.IGNORECASE,
)
_ACKNOWLEDGE = re.compile(
    r"\b(apologize|sorry|apologies|understand your frustration|deeply regret|"
    r"sincerely sorry|we regret|I understand|we understand|empathize)\b",
    re.IGNORECASE,
)


def _clamp_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (never exactly 0 or 1)."""
    if score <= 0:
        return MIN_REWARD
    if score >= 1:
        return MAX_REWARD
    if score < MIN_REWARD:
        return MIN_REWARD
    if score > MAX_REWARD:
        return MAX_REWARD
    return score


def _category_score(predicted: Optional[str], truth: str) -> float:
    """Returns score in (0, 1) - never exactly 0 or 1."""
    if not predicted:
        return MIN_REWARD
    is_correct = predicted.lower().strip() == truth.lower().strip()
    if is_correct:
        return MAX_REWARD
    return MIN_REWARD


def _priority_score(predicted: Optional[str], truth: str) -> float:
    """Returns score in (0, 1) - never exactly 0 or 1."""
    if not predicted:
        return MIN_REWARD
    
    pred = predicted.lower().strip()
    truth_lower = truth.lower().strip()
    
    if pred == truth_lower:
        return MAX_REWARD
    
    pred_rank = _PRIORITY_RANK.get(pred, -99)
    truth_rank = _PRIORITY_RANK.get(truth_lower, -99)
    diff = abs(pred_rank - truth_rank)
    
    if diff == 1:
        return 0.5
    
    return MIN_REWARD


def _action_score(predicted: Optional[str], truth: str) -> float:
    """Returns score in (0, 1) - never exactly 0 or 1."""
    if not predicted:
        return MIN_REWARD
    
    act = predicted.lower().strip()
    gt = truth.lower().strip()
    
    if act == gt:
        return MAX_REWARD
    
    near: Dict[str, set] = {
        "escalate": {"respond"},
        "respond": {"escalate"},
        "archive": {"respond"},
        "flag_spam": set(),
        "classify": {"respond", "archive"},
    }
    
    if act in near.get(gt, set()):
        return 0.4
    
    return MIN_REWARD


def _false_escalation_penalty(
    action_type: Optional[str], category: str, priority: str
) -> float:
    """Returns penalty (negative number or 0)."""
    if (action_type or "").lower() != "escalate":
        return 0.0
    
    if category in ("spam", "general"):
        return -0.30
    if category in ("billing", "tech") and priority in ("low", "medium"):
        return -0.20
    return 0.0


def _grade_response(
    response_text: Optional[str],
    config: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Multi-signal response quality grader.
    Returns (score in (0,1), breakdown_dict).
    """
    bd: Dict[str, float] = {
        "required_keywords": 0.0,
        "ideal_keywords":    0.0,
        "structure":         0.0,
        "acknowledgment":    0.0,
        "length":            0.0,
    }
    
    if not response_text or len(response_text.strip()) < 10:
        return MIN_REWARD, bd

    text = response_text.strip()
    tl = text.lower()
    words = len(text.split())

    # Required keywords (max 0.20)
    required = config.get("required_keywords", [])
    if required:
        required_score = sum(kw.lower() in tl for kw in required) / len(required)
        required_score = max(MIN_REWARD, min(MAX_REWARD, required_score))
        bd["required_keywords"] = 0.20 * required_score
    else:
        bd["required_keywords"] = 0.20

    # Ideal keywords (max 0.15)
    ideal = config.get("ideal_keywords", [])
    if ideal:
        ideal_score = sum(kw.lower() in tl for kw in ideal) / len(ideal)
        ideal_score = max(MIN_REWARD, min(MAX_REWARD, ideal_score))
        bd["ideal_keywords"] = 0.15 * ideal_score
    else:
        bd["ideal_keywords"] = 0.15

    # Structure: greeting + closing (max 0.10)
    bd["structure"] = (
        (0.05 if _GREETING.search(text) else MIN_REWARD) +
        (0.05 if _CLOSING.search(text) else MIN_REWARD)
    )

    # Acknowledgment/empathy (max 0.10)
    bd["acknowledgment"] = 0.10 if _ACKNOWLEDGE.search(text) else MIN_REWARD

    # Length (max 0.10)
    min_w: int = config.get("min_words", 40)
    if words >= min_w * 2:
        bd["length"] = 0.10
    elif words >= min_w:
        bd["length"] = 0.07
    elif words >= min_w // 2:
        bd["length"] = 0.03
    else:
        bd["length"] = MIN_REWARD

    raw = sum(bd.values())
    raw_score = raw / 0.65 if raw > 0 else MIN_REWARD
    raw_score = max(MIN_REWARD, min(MAX_REWARD, raw_score))
    
    return raw_score, bd


# ---------------------------------------------------------------------------
# Public entry point - THIS IS THE FUNCTION THAT WAS MISSING
# ---------------------------------------------------------------------------

def compute_reward(
    action: Any,
    ground_truth: Dict[str, Any],
    task: str = "email-classify",
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute per-step reward deterministically.
    Returns reward strictly between 0 and 1 (never exactly 0 or 1).
    """
    # Accept both Pydantic objects and plain dicts
    if isinstance(action, dict):
        class _Wrap:
            def __init__(self, d: dict) -> None:
                self.__dict__.update(d)
        action = _Wrap(action)

    # Core component scores
    predicted_cat = getattr(action, "category", None)
    predicted_pri = getattr(action, "priority", None)
    predicted_act = getattr(action, "action_type", None)
    predicted_res = getattr(action, "response", None)
    predicted_rea = getattr(action, "reasoning", None)

    true_cat = ground_truth.get("category", "")
    true_pri = ground_truth.get("priority", "")
    true_act = ground_truth.get("action_type", "")

    # Get individual scores (all already clamped to (0,1))
    cat = _category_score(predicted_cat, true_cat)
    pri = _priority_score(predicted_pri, true_pri)
    
    # Reasoning bonus - never exactly 1.0 or 0.0
    reas = MAX_REWARD if predicted_rea else MIN_REWARD
    resp = MAX_REWARD if predicted_res else MIN_REWARD

    # Wrong-category penalty
    wrong_cat_penalty = (
        -0.10
        if cat == MIN_REWARD and predicted_cat and predicted_cat.lower() in
           {"billing", "tech", "general", "complaint", "spam"}
        else 0.0
    )

    # Info dict for debugging
    info: Dict[str, Any] = {
        "ground_truth": {
            "category": true_cat,
            "priority": true_pri,
            "action_type": true_act,
        },
        "submitted": {
            "category": predicted_cat,
            "priority": predicted_pri,
            "action_type": predicted_act,
        },
        "category_score": cat,
        "wrong_category_penalty": wrong_cat_penalty,
        "priority_score": pri,
        "has_reasoning": bool(predicted_rea),
        "has_response": bool(predicted_res),
        "action_score": 0.0,
        "false_escalation_penalty": 0.0,
        "response_quality_score": 0.0,
        "response_quality_breakdown": {
            "required_keywords": 0.0,
            "ideal_keywords": 0.0,
            "structure": 0.0,
            "acknowledgment": 0.0,
            "length": 0.0,
        },
    }

    # ── Task 1: email-classify ──────────────────────────────────────────
    if task == "email-classify":
        reward = (
            0.40 * cat
            + wrong_cat_penalty
            + 0.30 * pri
            + 0.20 * reas
            + 0.10 * resp
        )

    # ── Task 2: email-triage ────────────────────────────────────────────
    elif task == "email-triage":
        act_s = _action_score(predicted_act, true_act)
        penalty = _false_escalation_penalty(predicted_act, true_cat, true_pri)
        reward = (
            0.20 * cat
            + wrong_cat_penalty
            + 0.15 * pri
            + 0.50 * act_s
            + 0.10 * reas
            + 0.05 * resp
            + penalty
        )
        info["action_score"] = act_s
        info["false_escalation_penalty"] = penalty

    # ── Task 3: email-respond ───────────────────────────────────────────
    elif task == "email-respond":
        act_s = _action_score(predicted_act, true_act)
        rq, rq_bd = _grade_response(predicted_res, ground_truth.get("response_config", {}))
        reward = (
            0.10 * cat
            + wrong_cat_penalty
            + 0.10 * pri
            + 0.10 * act_s
            + 0.65 * rq
            + 0.05 * reas
        )
        info["action_score"] = act_s
        info["response_quality_score"] = rq
        info["response_quality_breakdown"] = rq_bd

    else:
        reward = MIN_REWARD

    # FINAL CLAMP: Ensure reward is strictly between 0 and 1
    if reward <= 0:
        reward = MIN_REWARD
    elif reward >= 1:
        reward = MAX_REWARD
    
    if reward < MIN_REWARD:
        reward = MIN_REWARD
    if reward > MAX_REWARD:
        reward = MAX_REWARD
    
    # Round to 6 decimal places for consistency
    reward = reward * 0.95
    reward = max(0.01, min(0.99, reward))
    reward = float(f"{reward:.6f}")
    info["reward"] = reward
    
    return reward, info