"""
server/environment.py — EmailEnv simulation class.

Implements the full OpenEnv loop:
  reset()        → EmailObservation  (clean episode start)
  step(action)   → (EmailObservation, reward, done, info)
  state()        → EmailState        (full metadata snapshot)

Bug fixes applied
─────────────────
  • Action storage: uses action.model_dump(exclude_none=True) — never stores
    placeholder / schema-default values.
  • Input validation: step() validates action fields and surfaces errors in info
    instead of silently returning 0 reward with no diagnostic.
  • Determinism: no randomness anywhere; same input always produces same output.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.email_data import TASK_EMAILS
from app.models import EmailAction, EmailObservation, EmailState
from app.rewards import compute_reward

# ---------------------------------------------------------------------------
# Valid field values — used for input validation
# ---------------------------------------------------------------------------

_VALID_CATEGORIES  = {"billing", "tech", "general", "complaint", "spam"}
_VALID_PRIORITIES  = {"low", "medium", "high", "critical"}
_VALID_ACTION_TYPES = {"classify", "flag_spam", "archive", "escalate", "respond"}

_TASK_REQUIRED_ACTION: Dict[str, Optional[str]] = {
    "email-classify": "classify",   # must be 'classify'
    "email-triage":   None,         # any of the routing actions
    "email-respond":  None,         # escalate or respond
}

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

VALID_TASKS = ["email-classify", "email-triage", "email-respond"]

TASK_DESCRIPTIONS: Dict[str, str] = {
    "email-classify": (
        "EASY — Email Classification (5 emails)\n"
        "Classify each email by category and priority.\n"
        "action_type = 'classify'  (no routing needed)\n"
        "Scoring: category (40%) + priority (30%) + reasoning (20%) + response bonus (10%)"
    ),
    "email-triage": (
        "MEDIUM — Inbox Triage (7 emails)\n"
        "Classify each email and choose the correct routing action:\n"
        "  flag_spam | archive | escalate | respond\n"
        "Penalty: -0.30 for false escalation of non-urgent email.\n"
        "Scoring: category (20%) + priority (15%) + action (50%) + "
        "reasoning (10%) + response bonus (5%)"
    ),
    "email-respond": (
        "HARD — Response Generation (3 emails)\n"
        "Classify, route, and draft a professional response (≥ 80 words).\n"
        "Response graded on: required keywords, ideal coverage, "
        "professional structure, empathy, and length.\n"
        "Scoring: category (10%) + priority (10%) + action (10%) + "
        "response quality (65%) + reasoning (5%)"
    ),
}


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class EmailEnv:
    """
    Stateful email triage environment for one episode.

    The FastAPI server in server/app.py creates one EmailEnv per session ID
    and stores it in-memory.  Session TTL is 2 hours.
    """

    def __init__(self, task: str = "email-classify") -> None:
        if task not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task}'. Valid: {VALID_TASKS}")

        self.task       = task
        self.session_id = str(uuid.uuid4())

        # Initialised by reset()
        self._step:         int                  = 0
        self._done:         bool                 = False
        self._total_reward: float                = 0.0
        self._email_index:  int                  = 0
        self._emails:       List[Dict[str, Any]] = []
        self._history:      List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> EmailObservation:
        """Reset to a clean state and return the first observation."""
        self._step         = 0
        self._done         = False
        self._total_reward = 0.0
        self._email_index  = 0
        self._emails       = copy.deepcopy(TASK_EMAILS[self.task])
        self._history      = []
        return self._obs()

    def step(
        self, action: EmailAction
    ) -> Tuple[EmailObservation, float, bool, Dict[str, Any]]:

        # 🚨 Already finished safeguard
        if self._done:
            return self._obs(), 0.0, True, {
                "error": "Episode already finished. Call reset() to start a new episode."
            }

        # 🚨 NEW: Prevent index overflow BEFORE accessing email
        if self._email_index >= len(self._emails):
            self._done = True
            return self._obs(), 0.0, True, {
                "error": "No more emails to process."
            }

        # ── Input validation ────────────────────────────────────────────
        warnings: List[str] = []

        if action.category not in _VALID_CATEGORIES:
            warnings.append(
                f"Invalid category '{action.category}'. "
                f"Valid values: {sorted(_VALID_CATEGORIES)}"
            )

        if action.priority not in _VALID_PRIORITIES:
            warnings.append(
                f"Invalid priority '{action.priority}'. "
                f"Valid values: {sorted(_VALID_PRIORITIES)}"
            )

        if action.action_type and action.action_type not in _VALID_ACTION_TYPES:
            warnings.append(
                f"Invalid action_type '{action.action_type}'. "
                f"Valid values: {sorted(_VALID_ACTION_TYPES)}"
            )

        if self.task == "email-respond" and not action.response:
            warnings.append(
                "Hard task (email-respond) expects a 'response' field. "
                "Response quality is worth 65% of the reward."
            )

        # ✅ SAFE now
        self._step += 1
        email = self._emails[self._email_index]

        reward, info = compute_reward(action, email["ground_truth"], self.task)

        if warnings:
            info["validation_warnings"] = warnings

        # ✅ store real action
        action_record = action.model_dump(exclude_none=True)

        self._history.append({
            "step": self._step,
            "email_id": email["email_id"],
            "action": action_record,
            "reward": reward,
            "ground_truth": {
                "category": email["ground_truth"]["category"],
                "priority": email["ground_truth"]["priority"],
                "action_type": email["ground_truth"].get("action_type", ""),
            },
        })

        self._total_reward += reward
        self._email_index += 1

        # ✅ Correct done logic
        if self._email_index >= len(self._emails):
            self._done = True

        return self._obs(), reward, self._done, info

    def state(self) -> EmailState:
        """Return the full current episode state (for GET /state)."""
        total = len(self._emails)
        current_id = (
            self._emails[self._email_index]["email_id"]
            if not self._done and self._email_index < total
            else ""
        )
        return EmailState(
            ticket_id        = current_id,
            step_count       = self._step,
            task             = self.task,
            done             = self._done,
            total_reward     = round(self._total_reward, 4),
            emails_processed = self._email_index,
            emails_remaining = max(0, total - self._email_index),
            action_history   = self._history,
        )
    
    async def reset_async(self):
        return self.reset()

    async def step_async(self, action):
        return self.step(action)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> EmailObservation:
        total     = len(self._emails)
        remaining = max(0, total - self._email_index)
        desc      = TASK_DESCRIPTIONS[self.task]

        # Terminal observation
        if self._done or self._email_index >= total:
            n = total or 1
            summary = (
                f"Episode complete. {total} email(s) processed. "
                f"Total reward: {self._total_reward:.4f} / {float(total):.1f}. "
                f"Mean: {self._total_reward / n:.4f}"
            )
            return EmailObservation(
                message          = summary,
                history          = [],
                subject          = "[Inbox empty — episode complete]",
                sender           = "",
                timestamp        = "",
                email_id         = "",
                task             = self.task,
                task_description = desc,
                step             = self._step,
                emails_remaining = 0,
                total_emails     = total,
                inbox_size       = 0,
            )

        email = self._emails[self._email_index]
        history_strs = [
            f"[{m.get('timestamp','')}] {m.get('sender','')}:\n{m.get('body','')[:300]}"
            for m in email.get("thread_history", [])
        ]

        return EmailObservation(
            message          = email["body"],
            history          = history_strs,
            subject          = email["subject"],
            sender           = email["sender"],
            timestamp        = email["timestamp"],
            email_id         = email["email_id"],
            task             = self.task,
            task_description = desc,
            step             = self._step,
            emails_remaining = remaining,
            total_emails     = total,
            inbox_size       = remaining,
        )