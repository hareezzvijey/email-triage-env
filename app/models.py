"""
app/models.py — Typed Pydantic models for the Email Triage Environment.

Imports from openenv.core.env_server when available (production / HF Space).
Falls back to pure Pydantic so the repo works in offline / CI environments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# OpenEnv base-class shim — graceful fallback when openenv-core is absent
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import Action, Observation, State  # type: ignore
    _OPENENV_AVAILABLE = True
except ImportError:
    from pydantic import BaseModel

    class Action(BaseModel):       # type: ignore[no-redef]
        pass

    class Observation(BaseModel):  # type: ignore[no-redef]
        pass

    class State(BaseModel):        # type: ignore[no-redef]
        pass

    _OPENENV_AVAILABLE = False

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action / Observation / State
# ---------------------------------------------------------------------------

class EmailAction(Action):
    """Action submitted by the agent for each email.

    Required fields vary by task difficulty:
      easy   (email-classify) : category, priority
      medium (email-triage)   : category, priority, action_type
      hard   (email-respond)  : category, priority, action_type, response
    """
    category: str = Field(
        description="Email category: billing | tech | general | complaint | spam"
    )
    priority: str = Field(
        description="Urgency level: low | medium | high | critical"
    )
    action_type: Optional[str] = Field(
        default=None,
        description="Routing action: classify | flag_spam | archive | escalate | respond",
    )
    response: Optional[str] = Field(
        default=None,
        description="Full professional response draft (required for hard task, graded)",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's justification — earns bonus reward",
    )


class EmailObservation(Observation):
    """What the agent sees at each step — the current email plus episode context."""

    # OpenEnv spec fields
    message: str = Field(description="Full email body text")
    history: List[str] = Field(
        default_factory=list,
        description="Prior thread messages as formatted strings",
    )

    # Extended context fields
    subject: str        = Field(default="", description="Email subject line")
    sender: str         = Field(default="", description="Sender address")
    timestamp: str      = Field(default="", description="ISO-8601 timestamp")
    email_id: str       = Field(default="", description="Unique email identifier")
    task: str           = Field(default="", description="Active task identifier")
    task_description: str = Field(default="", description="Task instructions + scoring rubric")
    step: int           = Field(default=0,  description="Current step number")
    emails_remaining: int = Field(default=0, description="Emails left this episode")
    total_emails: int   = Field(default=0,  description="Total emails in episode")
    inbox_size: int     = Field(default=0,  description="Alias for emails_remaining")


class EmailState(State):
    """Full episode state — returned by GET /state."""

    # OpenEnv spec fields
    ticket_id: str  = Field(description="Current email being processed (email_id)")
    step_count: int = Field(description="Total steps taken so far")

    # Extended state fields
    task: str           = Field(default="",  description="Active task identifier")
    done: bool          = Field(default=False)
    total_reward: float = Field(default=0.0)
    emails_processed: int  = Field(default=0)
    emails_remaining: int  = Field(default=0)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP envelope models
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class GraderRequest(BaseModel):
    email_id: str
    task: str
    action: Dict[str, Any]

class GraderResult(BaseModel):
    email_id: str
    task: str
    action: Dict[str, Any]
    reward: float
    breakdown: Dict[str, Any]


class BaselineScore(BaseModel):
    task: str
    emails: int
    mean_reward: float
    min_reward: float
    max_reward: float
    scores: List[float]