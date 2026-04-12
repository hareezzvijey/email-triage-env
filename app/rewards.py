from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-6  # keeps values strictly inside (0,1)

def safe(x: float) -> float:
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


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _category_score(predicted: Optional[str], truth: str) -> float:
    if not predicted:
        return safe(0.0)
    return safe(1.0) if predicted.lower().strip() == truth.lower().strip() else safe(0.0)


def _priority_score(predicted: Optional[str], truth: str) -> float:
    if not predicted:
        return safe(0.0)

    pred = predicted.lower().strip()
    truth_lower = truth.lower().strip()

    if pred == truth_lower:
        return safe(1.0)

    pred_rank = _PRIORITY_RANK.get(pred, -99)
    truth_rank = _PRIORITY_RANK.get(truth_lower, -99)

    if abs(pred_rank - truth_rank) == 1:
        return safe(0.5)

    return safe(0.0)


def _action_score(predicted: Optional[str], truth: str) -> float:
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


# ---------------------------------------------------------------------------
# MULTIPLICATIVE PENALTY FUNCTIONS
# ---------------------------------------------------------------------------

def _get_wrong_category_factor(predicted_cat: Optional[str], true_cat: str) -> float:
    """Return 0.9 if wrong category, 1.0 if correct."""
    if not predicted_cat:
        return 1.0
    
    predicted_clean = predicted_cat.lower().strip()
    true_clean = true_cat.lower().strip()
    
    if predicted_clean == true_clean:
        return 1.0
    
    # ANY wrong category gets 10% penalty
    return 0.9


def _get_escalation_factor(
    action_type: Optional[str], category: str, priority: str
) -> float:
    """Return multiplier for escalation (1.0 = no penalty)."""
    if (action_type or "").lower() != "escalate":
        return 1.0

    if category in ("spam", "general"):
        return 0.7  # 30% penalty
    if category in ("billing", "tech") and priority in ("low", "medium"):
        return 0.8  # 20% penalty

    return 1.0


# ---------------------------------------------------------------------------
# Response grading
# ---------------------------------------------------------------------------

def _grade_response(
    response_text: Optional[str],
    config: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:

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

    required = config.get("required_keywords", [])
    if required:
        score = sum(kw.lower() in tl for kw in required) / len(required)
        bd["required_keywords"] = 0.20 * safe(score)
    else:
        bd["required_keywords"] = 0.20

    ideal = config.get("ideal_keywords", [])
    if ideal:
        score = sum(kw.lower() in tl for kw in ideal) / len(ideal)
        bd["ideal_keywords"] = 0.15 * safe(score)
    else:
        bd["ideal_keywords"] = 0.15

    structure_score = 0.0
    if _GREETING.search(text):
        structure_score += 0.05
    if _CLOSING.search(text):
        structure_score += 0.05
    bd["structure"] = EPS if structure_score == 0 else structure_score

    bd["acknowledgment"] = 0.10 if _ACKNOWLEDGE.search(text) else EPS

    min_w = config.get("min_words", 40)
    if words >= min_w * 2:
        bd["length"] = 0.10
    elif words >= min_w:
        bd["length"] = 0.07
    elif words >= min_w // 2:
        bd["length"] = 0.03
    else:
        bd["length"] = EPS

    raw = sum(bd.values())
    raw_score = raw / 0.65 if raw > 0 else EPS

    return safe(raw_score), bd


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def compute_reward(
    action: Any,
    ground_truth: Dict[str, Any],
    task: str = "email-classify",
) -> Tuple[float, Dict[str, Any]]:

    if isinstance(action, dict):
        class _Wrap:
            def __init__(self, d: dict):
                self.__dict__.update(d)
        action = _Wrap(action)

    predicted_cat = getattr(action, "category", None)
    predicted_pri = getattr(action, "priority", None)
    predicted_act = getattr(action, "action_type", None)
    predicted_res = getattr(action, "response", None)
    predicted_rea = getattr(action, "reasoning", None)

    true_cat = ground_truth.get("category", "")
    true_pri = ground_truth.get("priority", "")
    true_act = ground_truth.get("action_type", "")

    cat = _category_score(predicted_cat, true_cat)
    pri = _priority_score(predicted_pri, true_pri)
    reas = safe(1.0) if predicted_rea else safe(0.0)
    resp = safe(1.0) if predicted_res else safe(0.0)

    # Get multiplicative penalty factors
    wrong_cat_factor = _get_wrong_category_factor(predicted_cat, true_cat)
    escalation_factor = _get_escalation_factor(predicted_act, true_cat, true_pri)

    # Info dict
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
        "priority_score": pri,
        "has_reasoning": bool(predicted_rea),
        "has_response": bool(predicted_res),
        "wrong_category_factor": wrong_cat_factor,
        "escalation_factor": escalation_factor,
    }

    # ---------------- TASK 1: email-classify ----------------
    if task == "email-classify":
        base_reward = 0.40 * cat + 0.30 * pri + 0.20 * reas + 0.10 * resp
        reward = base_reward * wrong_cat_factor

    # ---------------- TASK 2: email-triage ----------------
    elif task == "email-triage":
        act_s = _action_score(predicted_act, true_act)
        base_reward = 0.20 * cat + 0.15 * pri + 0.50 * act_s + 0.10 * reas + 0.05 * resp
        reward = base_reward * escalation_factor * wrong_cat_factor
        info["action_score"] = act_s

    # ---------------- TASK 3: email-respond ----------------
    elif task == "email-respond":
        act_s = _action_score(predicted_act, true_act)
        rq, rq_bd = _grade_response(predicted_res, ground_truth.get("response_config", {}))
        base_reward = 0.10 * cat + 0.10 * pri + 0.10 * act_s + 0.65 * rq + 0.05 * reas
        reward = base_reward * wrong_cat_factor
        info["action_score"] = act_s
        info["response_quality_score"] = rq
        info["response_quality_breakdown"] = rq_bd

    else:
        reward = EPS

    # ========== FINAL CLAMP — strictly inside (0, 1) ==========
    # Never use round() or string formatting here: both can snap
    # edge values to exactly 0.0 or 1.0, failing Phase 2 validation.
    reward = max(EPS, min(1.0 - EPS, float(reward)))
    # ===========================================================

    info["reward"] = reward
    return reward, info