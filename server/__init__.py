"""
Email Triage Environment - Server components
"""

from server.app import create_fastapi_app
from server.environment import EmailEnv

__all__ = [
    "create_fastapi_app",
    "EmailEnv",
]