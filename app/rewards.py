"""
app/rewards.py — Deterministic reward computation for the Email Triage Environment.

All graders are fully deterministic — no LLM calls.

Reward breakdown by task
─────────────────────────
Task 1 · email-classify  (easy)
  category_score   0.40   exact match
  priority_score   0.30   exact=1.0, adjacent=0.5, wrong=0.0
  reasoning_bonus  0.20   any non-empty reasoning string
  response_bonus   0.10   optional; small bonus for including a draft

Task 2 · email-triage  (medium)
  category_score   0.20
  priority_score   0.15
  action_score     0.50   correct routing action
  reasoning_bonus  0.10
  response_bonus   0.05
  Penalty         -0.30   false escalation of non-urgent email

Task 3 · email-respond  (hard)
  category_score    0.10
  priority_score    0.10
  action_score      0.10
  response_quality  0.65   multi-signal (sum normalised to [0,1])
  reasoning_bonus   0.05
  Response quality sub-scores (raw max = 0.65):
    required_keywords  0.20
    ideal_keywords     0.15
    structure          0.10   greeting + closing
    acknowledgment     0.10   apology / empathy language
    length             0.10   word count vs min_words
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """
    Multi-signal response quality grader.
    Returns (normalised_score ∈ [0,1], breakdown_dict).
    Raw sub-scores sum to at most 0.65; we normalise by dividing by 0.65.
    """
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

    raw = sum(bd.values())          # max = 0.65
    return min(1.0, raw / 0.65), bd


# ---------------------------------------------------------------------------
# Public entry point  (signature matches the build prompt spec)
# ---------------------------------------------------------------------------

def compute_reward(
    action: Any,                    # EmailAction object OR plain dict
    ground_truth: Dict[str, Any],
    task: str = "email-classify",
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute per-step reward deterministically.

    Parameters
    ----------
    action       : EmailAction (pydantic) or dict with same keys
    ground_truth : email ground_truth sub-dict from email_data.py
    task         : email-classify | email-triage | email-respond

    Returns
    -------
    reward : float in [0.0, 1.0]
    info   : dict with full scoring breakdown
    """
    # Accept both pydantic objects and plain dicts
    if isinstance(action, dict):
        class _Wrap:
            def __init__(self, d: dict) -> None:
                self.__dict__.update(d)
        action = _Wrap(action)

    cat  = _category_score(getattr(action, "category",    None), ground_truth["category"])
    pri  = _priority_score(getattr(action, "priority",    None), ground_truth["priority"])
    reas = 1.0 if getattr(action, "reasoning", None) else 0.0
    resp = 1.0 if getattr(action, "response",  None) else 0.0

    info: Dict[str, Any] = {
        "ground_truth": {
            "category":    ground_truth["category"],
            "priority":    ground_truth["priority"],
            "action_type": ground_truth.get("action_type", ""),
        },
        "category_score": cat,
        "priority_score": pri,
        "has_reasoning":  bool(reas),
        "has_response":   bool(resp),
    }

    # ── Task 1: email-classify ──────────────────────────────────────────
    if task == "email-classify":
        reward = 0.40 * cat + 0.30 * pri + 0.20 * reas + 0.10 * resp
        info.update(action_score=None, response_quality_score=None)

    # ── Task 2: email-triage ────────────────────────────────────────────
    elif task == "email-triage":
        act     = _action_score(getattr(action, "action_type", None), ground_truth.get("action_type", ""))
        penalty = _false_escalation_penalty(
            getattr(action, "action_type", None),
            ground_truth["category"],
            ground_truth["priority"],
        )
        reward = 0.20 * cat + 0.15 * pri + 0.50 * act + 0.10 * reas + 0.05 * resp + penalty
        info.update(action_score=act, false_escalation_penalty=penalty, response_quality_score=None)

    # ── Task 3: email-respond ───────────────────────────────────────────
    elif task == "email-respond":
        act = _action_score(getattr(action, "action_type", None), ground_truth.get("action_type", ""))
        rq, rq_bd = _grade_response(getattr(action, "response", None), ground_truth.get("response_config", {}))
        reward = 0.10 * cat + 0.10 * pri + 0.10 * act + 0.65 * rq + 0.05 * reas
        info.update(action_score=act, response_quality_score=rq, response_quality_breakdown=rq_bd)

    else:
        reward = 0.0

    reward = round(max(0.0, min(1.0, reward)), 4)
    info["reward"] = reward
    return reward, info