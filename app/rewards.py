from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-6  # 0.000001 - keeps values strictly away from boundaries

def safe(x: float) -> float:
    """Ensure value is strictly between 0 and 1."""
    return max(EPS, min(1.0 - EPS, x))

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


def _category_score(predicted: Optional[str], truth: str) -> float:
    """Returns safe(1.0) if correct, safe(0.0) if wrong."""
    if not predicted:
        return safe(0.0)
    is_correct = predicted.lower().strip() == truth.lower().strip()
    return safe(1.0) if is_correct else safe(0.0)


def _priority_score(predicted: Optional[str], truth: str) -> float:
    """Returns safe(1.0) if exact, safe(0.5) if adjacent, safe(0.0) otherwise."""
    if not predicted:
        return safe(0.0)
    
    pred = predicted.lower().strip()
    truth_lower = truth.lower().strip()
    
    if pred == truth_lower:
        return safe(1.0)
    
    pred_rank = _PRIORITY_RANK.get(pred, -99)
    truth_rank = _PRIORITY_RANK.get(truth_lower, -99)
    diff = abs(pred_rank - truth_rank)
    
    if diff == 1:
        return safe(0.5)
    
    return safe(0.0)


def _action_score(predicted: Optional[str], truth: str) -> float:
    """Returns safe(1.0) if exact, safe(0.4) if near, safe(0.0) otherwise."""
    if not predicted:
        return safe(0.0)
    
    act = predicted.lower().strip()
    gt = truth.lower().strip()
    
    if act == gt:
        return safe(1.0)
    
    near: Dict[str, set] = {
        "escalate": {"respond"},
        "respond": {"escalate"},
        "archive": {"respond"},
        "flag_spam": set(),
        "classify": {"respond", "archive"},
    }
    
    if act in near.get(gt, set()):
        return safe(0.4)
    
    return safe(0.0)


def _false_escalation_penalty(
    action_type: Optional[str], category: str, priority: str
) -> float:
    """Returns penalty (negative number or 0)."""
    if (action_type or "").lower() != "escalate":
        return EPS
    
    if category in ("spam", "general"):
        return -0.30
    if category in ("billing", "tech") and priority in ("low", "medium"):
        return -0.20
    return EPS


def _grade_response(
    response_text: Optional[str],
    config: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Multi-signal response quality grader.
    Returns (score in (0,1), breakdown_dict).
    FIXED: Never returns 0.0 - uses EPS for minimum.
    """
    # Initialize breakdown with EPS 
    bd: Dict[str, float] = {
        "required_keywords": EPS,
        "ideal_keywords": EPS,
        "structure": EPS,
        "acknowledgment": EPS,
        "length": EPS,
    }
    
    if not response_text or len(response_text.strip()) < 10:
        return safe(0.0), bd

    text = response_text.strip()
    tl = text.lower()
    words = len(text.split())

    # Required keywords (max 0.20)
    required = config.get("required_keywords", [])
    if required:
        required_score = sum(kw.lower() in tl for kw in required) / len(required)
        bd["required_keywords"] = 0.20 * max(EPS, min(1.0 - EPS, required_score))
    else:
        bd["required_keywords"] = 0.20

    # Ideal keywords (max 0.15)
    ideal = config.get("ideal_keywords", [])
    if ideal:
        ideal_score = sum(kw.lower() in tl for kw in ideal) / len(ideal)
        bd["ideal_keywords"] = 0.15 * max(EPS, min(1.0 - EPS, ideal_score))
    else:
        bd["ideal_keywords"] = 0.15

    # Structure: greeting + closing (max 0.10)
    structure_score = 0.0
    if _GREETING.search(text):
        structure_score += 0.05
    if _CLOSING.search(text):
        structure_score += 0.05
    # Use EPS only when genuinely zero, otherwise use actual score
    bd["structure"] = EPS if structure_score == 0 else structure_score

    # Acknowledgment/empathy (max 0.10)
    bd["acknowledgment"] = 0.10 if _ACKNOWLEDGE.search(text) else EPS

    # Length (max 0.10)
    min_w: int = config.get("min_words", 40)
    if words >= min_w * 2:
        bd["length"] = 0.10
    elif words >= min_w:
        bd["length"] = 0.07
    elif words >= min_w // 2:
        bd["length"] = 0.03
    else:
        bd["length"] = EPS

    raw = sum(bd.values())
    
    # CRITICAL FIX: Handle raw == 0 case
    if raw <= 0:
        raw_score = EPS
    else:
        raw_score = min(1.0 - EPS, raw / 0.65)
    
    return safe(raw_score), bd


# ---------------------------------------------------------------------------
# Public entry point
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

    # Get individual scores (all already safe-clamped)
    cat = _category_score(predicted_cat, true_cat)
    pri = _priority_score(predicted_pri, true_pri)
    
    # Reasoning and response bonuses
    reas = safe(1.0) if predicted_rea else safe(0.0)
    resp = safe(1.0) if predicted_res else safe(0.0)

    # Wrong-category penalty - using direct string comparison
    wrong_cat_penalty = EPS
    if predicted_cat and predicted_cat.lower() != true_cat.lower():
        if predicted_cat.lower() in {"billing", "tech", "general", "complaint", "spam"}:
            wrong_cat_penalty = -0.10

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
        "action_score": EPS,
        "false_escalation_penalty": EPS,
        "response_quality_score": EPS,
        "response_quality_breakdown": {
            "required_keywords": EPS,
            "ideal_keywords": EPS,
            "structure": EPS,
            "acknowledgment": EPS,
            "length": EPS,
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
        reward = safe(0.0)
    
    # Clamping
    reward = safe(reward)
    reward = float(f"{reward:.6f}")
    info["reward"] = reward
    
    return reward, info