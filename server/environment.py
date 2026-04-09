"""
server/environment.py — EmailEnv simulation class.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Tuple

from app.email_data import TASK_EMAILS
from app.models import EmailAction, EmailObservation, EmailState
from app.rewards import compute_reward

VALID_TASKS = ["email-classify", "email-triage", "email-respond"]

TASK_DESCRIPTIONS = {
    "email-classify": "EASY — Email Classification (5 emails)\nClassify each email by category and priority.\naction_type = 'classify' (no routing needed)\nScoring: category (40%) + priority (30%) + reasoning (20%) + response bonus (10%)",
    "email-triage": "MEDIUM — Inbox Triage (7 emails)\nClassify each email and choose the correct routing action:\n  flag_spam | archive | escalate | respond\nPenalty: -0.30 for false escalation of non-urgent email.\nScoring: category (20%) + priority (15%) + action (50%) + reasoning (10%) + response bonus (5%)",
    "email-respond": "HARD — Response Generation (3 emails)\nClassify, route, and draft a professional response (≥ 80 words).\nResponse graded on: required keywords, ideal coverage, professional structure, empathy, and length.\nScoring: category (10%) + priority (10%) + action (10%) + response quality (65%) + reasoning (5%)",
}

class EmailEnv:
    def __init__(self, task: str = "email-classify"):
        if task not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task}'. Valid: {VALID_TASKS}")
        self.task = task
        self._step = 0
        self._done = False
        self._total_reward = 0.0
        self._email_index = 0
        self._emails = []
        self._history = []

    def reset(self) -> EmailObservation:
        self._step = 0
        self._done = False
        self._total_reward = 0.0
        self._email_index = 0
        self._emails = copy.deepcopy(TASK_EMAILS[self.task])
        self._history = []
        return self._obs()

    def step(self, action: EmailAction) -> Tuple[EmailObservation, float, bool, Dict[str, Any]]:
        """Process action and return (observation, reward, done, info)."""
        if self._done:
            return self._obs(), 0.0001, True, {"error": "Episode already finished"}
        
        self._step += 1
        email = self._emails[self._email_index]
        
        # Convert action to dict for compute_reward if needed
        action_dict = action.model_dump(exclude_none=True) if hasattr(action, 'model_dump') else action.__dict__
        reward, info = compute_reward(action_dict, email["ground_truth"], self.task)
        
        self._history.append({
            "step": self._step,
            "email_id": email["email_id"],
            "action": action_dict,
            "reward": reward,
            "ground_truth": {
                "category": email["ground_truth"]["category"],
                "priority": email["ground_truth"]["priority"],
                "action_type": email["ground_truth"].get("action_type", ""),
            },
        })
        self._total_reward += reward
        self._email_index += 1
        
        if self._email_index >= len(self._emails):
            self._done = True
        
        return self._obs(), reward, self._done, info

    def state(self) -> EmailState:
        total = len(self._emails)
        current_id = (
            self._emails[self._email_index]["email_id"]
            if not self._done and self._email_index < total
            else ""
        )
        return EmailState(
            ticket_id=current_id,
            step_count=self._step,
            task=self.task,
            done=self._done,
            total_reward=round(self._total_reward, 6),
            emails_processed=self._email_index,
            emails_remaining=max(0, total - self._email_index),
            action_history=self._history,
        )

    def _obs(self) -> EmailObservation:
        total = len(self._emails)
        remaining = max(0, total - self._email_index)
        desc = TASK_DESCRIPTIONS[self.task]

        if self._done or self._email_index >= total:
            summary = f"Episode complete. {total} email(s) processed."
            return EmailObservation(
                message=summary,
                history=[],
                subject="[Inbox empty — episode complete]",
                sender="",
                timestamp="",
                email_id="",
                task=self.task,
                task_description=desc,
                step=self._step,
                emails_remaining=0,
                total_emails=total,
                inbox_size=0,
            )

        email = self._emails[self._email_index]
        history_strs = [
            f"[{m.get('timestamp','')}] {m.get('sender','')}:\n{m.get('body','')[:300]}"
            for m in email.get("thread_history", [])
        ]

        return EmailObservation(
            message=email["body"],
            history=history_strs,
            subject=email["subject"],
            sender=email["sender"],
            timestamp=email["timestamp"],
            email_id=email["email_id"],
            task=self.task,
            task_description=desc,
            step=self._step,
            emails_remaining=remaining,
            total_emails=total,
            inbox_size=remaining,
        )