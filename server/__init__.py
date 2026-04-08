"""
Email Triage Environment - Server components
"""

# Bug 1/3 fix: import from server.environment (the actual file),
# NOT from server.email_triage_env_new_environment (stale name that never existed).
from server.environment import EmailEnv, VALID_TASKS, TASK_DESCRIPTIONS
from server.app import app

__all__ = [
    "app",
    "EmailEnv",
    "VALID_TASKS",
    "TASK_DESCRIPTIONS",
]