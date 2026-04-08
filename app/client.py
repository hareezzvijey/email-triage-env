"""
app/client.py — Async HTTP client for the Email Triage Environment.

Extends openenv.core.env_client.EnvClient when available.
Falls back to a pure-httpx implementation with the identical public interface.

Usage (remote server)
----------------------
    from app.client import EmailTriageEnvClient
    from app.models import EmailAction

    env = await EmailTriageEnvClient.from_url("https://your-space.hf.space")
    obs = await env.reset(task="email-triage")
    while not done:
        result = await env.step(action)
        done = result.done
    await env.close()

Usage (Docker image — auto spins up container)
-----------------------------------------------
    env = await EmailTriageEnvClient.from_docker_image("email-triage-env:latest")
    ...
"""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from typing import Any, Dict, Optional

import httpx

from app.models import EmailAction, EmailObservation, EmailState, StepResult

_DEFAULT_TIMEOUT = 60.0
_STARTUP_WAIT    = 5      # seconds to wait for Docker container to start

# ---------------------------------------------------------------------------
# Optional openenv-core base class
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_client import EnvClient as _Base  # type: ignore
    _HAVE_OPENENV = True
except ImportError:
    _Base = object           # type: ignore[misc,assignment]
    _HAVE_OPENENV = False


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class EmailTriageEnvClient(_Base):  # type: ignore[misc]
    """
    Async HTTP client wrapping the Email Triage REST API.

    Implements the three methods required by openenv.core.env_client.EnvClient:
      _step_payload(action)  → dict  posted to POST /step
      _parse_result(data)    → StepResult from /step JSON response
      _parse_state(data)     → EmailState from /state JSON response
    """

    def __init__(self, base_url: str, task: str = "email-classify") -> None:
        self.base_url   = base_url.rstrip("/")
        self.task       = task
        self.session_id = str(uuid.uuid4())
        self._container_name: Optional[str] = None
        self._http: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    async def from_url(
        cls, url: str, task: str = "email-classify"
    ) -> "EmailTriageEnvClient":
        """Connect to an already-running environment server."""
        client = cls(base_url=url, task=task)
        await client._ensure_http()
        return client

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        task: str = "email-classify",
        host_port: int = 7860,
    ) -> "EmailTriageEnvClient":
        """Launch a local Docker container and connect to it."""
        name = f"email-triage-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["docker", "run", "-d", "--name", name,
             "-p", f"{host_port}:7860", image_name],
            check=True, capture_output=True,
        )
        await asyncio.sleep(_STARTUP_WAIT)
        client = cls(base_url=f"http://localhost:{host_port}", task=task)
        client._container_name = name
        await client._ensure_http()
        return client

    # ------------------------------------------------------------------
    # openenv.core.env_client.EnvClient interface
    # ------------------------------------------------------------------

    def _step_payload(self, action: EmailAction) -> Dict[str, Any]:
        """Serialise EmailAction → dict for POST /step."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, data: Dict[str, Any]) -> StepResult:
        """Parse POST /step JSON → StepResult."""
        return StepResult(
            observation=EmailObservation(**data["observation"]),
            reward=float(data["reward"]),
            done=bool(data["done"]),
            info=data.get("info", {}),
        )

    def _parse_state(self, data: Dict[str, Any]) -> EmailState:
        """Parse GET /state JSON → EmailState."""
        return EmailState(**data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reset(self, task: Optional[str] = None) -> EmailObservation:
        """Reset the environment (optionally switch task)."""
        if task:
            self.task = task
        self.session_id = str(uuid.uuid4())
        http = await self._ensure_http()
        resp = await http.post(
            f"{self.base_url}/reset",
            json={"task": self.task},
            headers={"X-Session-ID": self.session_id},
        )
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data.get("session_id", self.session_id)
        return EmailObservation(**data["observation"])

    async def step(self, action: EmailAction) -> StepResult:
        """Submit one action, receive next observation + reward."""
        http = await self._ensure_http()
        resp = await http.post(
            f"{self.base_url}/step",
            json=self._step_payload(action),
            headers={"X-Session-ID": self.session_id},
        )
        resp.raise_for_status()
        return self._parse_result(resp.json())

    async def state(self) -> EmailState:
        """Return the full current episode state."""
        http = await self._ensure_http()
        resp = await http.get(
            f"{self.base_url}/state",
            headers={"X-Session-ID": self.session_id},
        )
        resp.raise_for_status()
        return self._parse_state(resp.json())

    async def close(self) -> None:
        """Shut down HTTP client and (if applicable) the Docker container."""
        if self._http:
            await self._http.aclose()
            self._http = None
        if self._container_name:
            subprocess.run(["docker", "stop", self._container_name],
                           check=False, capture_output=True)
            subprocess.run(["docker", "rm",   self._container_name],
                           check=False, capture_output=True)
            self._container_name = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "EmailTriageEnvClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
        return self._http