"""
app/rewards.py — Deterministic reward computation for the Email Triage Environment.

Boundary fix
─────────────
Validator requires scores strictly in (0, 1) — never exactly 0.0 or 1.0.

Root causes found:
  • Exactly 1.0: perfect email-classify (cat+pri+reas+resp) = 0.40+0.30+0.20+0.10 = 1.0
  • Exactly 0.0: wrong category + wrong_cat_penalty (-0.10) + no reasoning = -0.10 → 0.0

Fix: _boundary_clamp(x) = max(1e-4, min(0.9999, x)) applied ONCE in compute_reward().
DO NOT multiply reward by 0.95 — that distorts the signal and causes double-shrinkage
when the grader endpoint naively applies it again.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

_EPS = 1e-4   # 0.0001 — keeps every reward strictly inside (0, 1)

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


def _boundary_clamp(value: float) -> float:
    """Return value clamped to the open interval (0, 1).
    Result is always in [_EPS, 1 - _EPS] = [0.0001, 0.9999].
    Applied ONCE at the end of compute_reward(). Never call this in callers.
    """
    return round(max(_EPS, min(1.0 - _EPS, value)), 6)


def _category_score(predicted: Optional[str], truth: str) -> float:
    if not predicted:
        return 0.0
    return 1.0 if predicted.lower().strip() == truth.lower().strip() else 0.0


def _priority_score(predicted: Optional[str], truth: str) -> float:
    if not predicted:
        return 0.0
    pred = predicted.lower().strip()
    if pred == truth:
        return 1.0
    diff = abs(_PRIORITY_RANK.get(pred, -99) - _PRIORITY_RANK.get(truth, -99))
    return 0.5 if diff == 1 else 0.0


def _action_score(predicted: Optional[str], truth: str) -> float:
    if not predicted:
        return 0.0
    act = predicted.lower().strip()
    gt  = truth.lower().strip()
    if act == gt:
        return 1.0
    near: Dict[str, set] = {
        "escalate": {"respond"},
        "respond":  {"escalate"},
        "archive":  {"respond"},
        "flag_spam": set(),
        "classify":  {"respond", "archive"},
    }
    return 0.4 if act in near.get(gt, set()) else 0.0


def _false_escalation_penalty(
    action_type: Optional[str], category: str, priority: str
) -> float:
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
    bd: Dict[str, float] = {
        "required_keywords": 0.0,
        "ideal_keywords":    0.0,
        "structure":         0.0,
        "acknowledgment":    0.0,
        "length":            0.0,
    }
    if not response_text or len(response_text.strip()) < 10:
        return 0.0, bd

    text  = response_text.strip()
    tl    = text.lower()
    words = len(text.split())

    required = config.get("required_keywords", [])
    bd["required_keywords"] = (
        0.20 * (sum(kw.lower() in tl for kw in required) / len(required))
        if required else 0.20
    )

    ideal = config.get("ideal_keywords", [])
    bd["ideal_keywords"] = (
        0.15 * min(1.0, sum(kw.lower() in tl for kw in ideal) / len(ideal))
        if ideal else 0.15
    )

    bd["structure"] = (
        (0.05 if _GREETING.search(text) else 0.0) +
        (0.05 if _CLOSING.search(text) else 0.0)
    )
    bd["acknowledgment"] = 0.10 if _ACKNOWLEDGE.search(text) else 0.0

    min_w: int = config.get("min_words", 40)
    bd["length"] = (
        0.10 if words >= min_w * 2 else
        0.07 if words >= min_w      else
        0.03 if words >= min_w // 2 else
        0.0
    )

    raw = sum(bd.values())
    return min(1.0, raw / 0.65), bd


def compute_reward(
    action: Any,
    ground_truth: Dict[str, Any],
    task: str = "email-classify",
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute per-step reward deterministically.

    Returns reward strictly in (0, 1) via _boundary_clamp applied ONCE here.
    Callers (e.g. /grader, /baseline) must NOT apply additional clamping or
    multiply the returned value — doing so will distort the reward signal.
    """
    if isinstance(action, dict):
        class _Wrap:
            def __init__(self, d: dict) -> None:
                self.__dict__.update(d)
        action = _Wrap(action)

    predicted_cat = getattr(action, "category",    None)
    predicted_pri = getattr(action, "priority",    None)
    predicted_act = getattr(action, "action_type", None)
    predicted_res = getattr(action, "response",    None)
    predicted_rea = getattr(action, "reasoning",   None)

    true_cat = ground_truth.get("category", "")
    true_pri = ground_truth.get("priority", "")
    true_act = ground_truth.get("action_type", "")

    cat  = _category_score(predicted_cat, true_cat)
    pri  = _priority_score(predicted_pri, true_pri)
    reas = 1.0 if predicted_rea else 0.0
    resp = 1.0 if predicted_res else 0.0

    # Wrong-category penalty (cat==0.0 and it's a valid but wrong category)
    wrong_cat_penalty = (
        -0.10
        if cat == 0.0 and predicted_cat and predicted_cat.lower() in
           {"billing", "tech", "general", "complaint", "spam"}
        else 0.0
    )

    info: Dict[str, Any] = {
        "ground_truth": {
            "category":    true_cat,
            "priority":    true_pri,
            "action_type": true_act,
        },
        "submitted": {
            "category":    predicted_cat,
            "priority":    predicted_pri,
            "action_type": predicted_act,
        },
        "category_score":             cat,
        "wrong_category_penalty":     wrong_cat_penalty,
        "priority_score":             pri,
        "has_reasoning":              bool(reas),
        "has_response":               bool(resp),
        "action_score":               0.0,
        "false_escalation_penalty":   0.0,
        "response_quality_score":     0.0,
        "response_quality_breakdown": {
            "required_keywords": 0.0,
            "ideal_keywords":    0.0,
            "structure":         0.0,
            "acknowledgment":    0.0,
            "length":            0.0,
        },
    }

    if task == "email-classify":
        raw_reward = (
            0.40 * cat + wrong_cat_penalty +
            0.30 * pri + 0.20 * reas + 0.10 * resp
        )

    elif task == "email-triage":
        act_s   = _action_score(predicted_act, true_act)
        penalty = _false_escalation_penalty(predicted_act, true_cat, true_pri)
        raw_reward = (
            0.20 * cat + wrong_cat_penalty + 0.15 * pri +
            0.50 * act_s + 0.10 * reas + 0.05 * resp + penalty
        )
        info["action_score"]             = act_s
        info["false_escalation_penalty"] = penalty

    elif task == "email-respond":
        act_s    = _action_score(predicted_act, true_act)
        rq, rq_bd = _grade_response(predicted_res, ground_truth.get("response_config", {}))
        raw_reward = (
            0.10 * cat + wrong_cat_penalty + 0.10 * pri +
            0.10 * act_s + 0.65 * rq + 0.05 * reas
        )
        info["action_score"]               = act_s
        info["response_quality_score"]     = rq
        info["response_quality_breakdown"] = rq_bd

    else:
        raw_reward = 0.0

    # Apply boundary clamp ONCE — result is always in [0.0001, 0.9999]
    reward = _boundary_clamp(raw_reward)
    info["reward"] = reward
    return reward, info