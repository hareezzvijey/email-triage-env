"""
inference.py — Baseline LLM agent for the Email Triage Environment
==================================================================

Mandatory environment variables
---------------------------------
  API_BASE_URL            LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME              Model name    (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN                API key

Optional environment variables
---------------------------------
  EMAIL_TRIAGE_SERVER_URL URL of running env server (default: http://localhost:7860)
  LOCAL_IMAGE_NAME        Docker image — auto-spins up container if set
  EMAIL_TRIAGE_TASK       Run one task instead of all three

Stdout format  (mandatory — do NOT change field names or order)
---------------------------------------------------------------
  [START] task=<task> env=email-triage-env model=<model>
  [STEP]  step=<n> action=<summary> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from app.client import EmailTriageEnvClient
from app.models import EmailAction, EmailObservation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str       = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str         = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY: str            = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"
SERVER_URL: str         = os.getenv("EMAIL_TRIAGE_SERVER_URL", "http://localhost:7860")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")
SINGLE_TASK: Optional[str]      = os.getenv("EMAIL_TRIAGE_TASK")

ALL_TASKS         = ["email-classify", "email-triage", "email-respond"]
BENCHMARK         = "email-triage-env"
MAX_STEPS         = 10
TEMPERATURE       = 0.2
MAX_TOKENS        = 800
SUCCESS_THRESHOLD = 0.50   # episode mean reward ≥ this → success

# ---------------------------------------------------------------------------
# Per-task system prompts
# ---------------------------------------------------------------------------

_SYSTEM: Dict[str, str] = {

    "email-classify": textwrap.dedent("""
        You are an expert email classifier. For each email output ONLY a JSON object
        with no markdown or extra text:

        {
          "category":  "<billing|tech|general|complaint|spam>",
          "priority":  "<low|medium|high|critical>",
          "reasoning": "<one sentence justification>"
        }

        Category rules:
          billing   — payment, invoice, refund, order, subscription
          tech      — software bugs, installation, crashes, API, error codes
          spam      — promotions, phishing, scams, unsolicited bulk mail
          complaint — dissatisfaction, threats, repeat failures
          general   — internal comms, social, low-stakes queries

        Priority rules:
          critical — immediate action (legal/financial/safety risk, outages)
          high     — same-day attention (unhappy paying customer, demo tomorrow)
          medium   — 1–2 day response window
          low      — no time pressure
    """).strip(),

    "email-triage": textwrap.dedent("""
        You are an expert email triage agent. Output ONLY a JSON object:

        {
          "category":    "<billing|tech|general|complaint|spam>",
          "priority":    "<low|medium|high|critical>",
          "action_type": "<flag_spam|archive|escalate|respond>",
          "reasoning":   "<one sentence justifying category, priority, and action>"
        }

        Action rules:
          flag_spam — spam / phishing / scam (never respond)
          archive   — low-priority general / social messages that need no reply
          escalate  — critical alerts, legal deadlines, enterprise crises
          respond   — billing / tech inquiries that need a human reply

        WARNING: Only escalate if priority is high or critical AND category is
        complaint, urgent tech, or there is a legal/financial threat.
        Escalating spam or general emails incurs a -0.30 score penalty.
    """).strip(),

    "email-respond": textwrap.dedent("""
        You are a senior customer support agent. Output ONLY a JSON object:

        {
          "category":    "<billing|tech|general|complaint|spam>",
          "priority":    "<low|medium|high|critical>",
          "action_type": "<escalate|respond>",
          "response":    "<full professional response — minimum 80 words>",
          "reasoning":   "<brief explanation>"
        }

        Response quality is graded on:
          1. Opens with a greeting (Dear / Hello)
          2. Sincere apology / acknowledgment
          3. Addresses EVERY specific concern raised
          4. Clear next steps and timeline
          5. Professional closing (Kind regards / Sincerely)
          6. At least 80 words (short responses are penalised)

        Use action_type="escalate" for critical complaints needing senior management,
        but always include a full acknowledgment in response.
    """).strip(),
}

# ---------------------------------------------------------------------------
# Logging helpers  (mandatory stdout format — do NOT modify)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(
    step: int, action_summary: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_summary} "
        f"reward={reward:.6f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _user_prompt(obs: EmailObservation, step: int) -> str:
    thread = ""
    if obs.history:
        thread = "\n\nThread history:\n" + "\n".join(f"  {m}" for m in obs.history[-3:])

    return textwrap.dedent(f"""
        Step {step} | Emails remaining: {obs.emails_remaining}

        ── INCOMING EMAIL ──────────────────────────────────────────
        From    : {obs.sender}
        Subject : {obs.subject}
        Received: {obs.timestamp}
        {thread}

        Message:
        {obs.message}
        ────────────────────────────────────────────────────────────

        Output your JSON action object now.
    """).strip()


def _call_llm(llm: OpenAI, task: str, obs: EmailObservation, step: int) -> str:
    try:
        r = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM[task]},
                {"role": "user",   "content": _user_prompt(obs, step)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return "{}"


def _parse_action(raw: str, task: str) -> Tuple[EmailAction, Optional[str]]:
    """Parse LLM JSON output → EmailAction with safe fallback defaults."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    # Extract JSON object
    s, e = text.find("{"), text.rfind("}") + 1
    if s != -1 and e > s:
        text = text[s:e]

    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        defaults: Dict[str, EmailAction] = {
            "email-classify": EmailAction(category="general", priority="low",
                                          reasoning="JSON parse error"),
            "email-triage":   EmailAction(category="general", priority="low",
                                          action_type="archive", reasoning="JSON parse error"),
            "email-respond":  EmailAction(
                category="general", priority="low", action_type="respond",
                response=(
                    "Dear Customer, I sincerely apologize for the inconvenience. "
                    "We are treating this as a top priority and will resolve it urgently. "
                    "Kind regards, Support Team"
                ),
                reasoning="JSON parse error",
            ),
        }
        return defaults[task], f"JSONDecodeError: {exc}"

    valid: Dict[str, set] = {
        "email-classify": {"classify"},
        "email-triage":   {"flag_spam", "archive", "escalate", "respond"},
        "email-respond":  {"escalate", "respond"},
    }
    act_type = d.get("action_type", "")
    if task == "email-classify":
        act_type = "classify"
    elif act_type not in valid.get(task, set()):
        act_type = list(valid[task])[0]

    return EmailAction(
        category    = d.get("category",  "general"),
        priority    = d.get("priority",  "low"),
        action_type = act_type or None,
        response    = d.get("response"),
        reasoning   = d.get("reasoning"),
    ), None


def _summary(action: EmailAction) -> str:
    parts = [f"cat={action.category}", f"pri={action.priority}"]
    if action.action_type:
        parts.append(f"act={action.action_type}")
    if action.response:
        parts.append(f"resp={action.response[:30].replace(chr(10),' ')!r}...")
    return "(" + ",".join(parts) + ")"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(env: EmailTriageEnvClient, task: str, llm: OpenAI) -> None:
    """Run one full episode and emit mandatory [START]/[STEP]/[END] logs."""
    log_start(task=task, model=MODEL_NAME)
    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False

    try:
        # ── Core RL loop ────────────────────────────────────────────────
        # obs = env.reset()
        # while not done:
        #     action = agent(obs)
        #     obs, reward, done, _ = env.step(action)
        # ────────────────────────────────────────────────────────────────
        obs  = await env.reset(task=task)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            raw    = _call_llm(llm, task, obs, step)
            action, parse_err = _parse_action(raw, task)

            result    = await env.step(action)
            obs       = result.observation
            reward    = result.reward
            done      = result.done
            api_error = result.info.get("error") or parse_err

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action_summary=_summary(action),
                     reward=reward, done=done, error=api_error)

        n       = len(rewards)
        mean    = sum(rewards) / n if n else 0.0
        success = mean >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    llm   = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = [SINGLE_TASK] if SINGLE_TASK else ALL_TASKS

    env = (
        await EmailTriageEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
        if LOCAL_IMAGE_NAME
        else await EmailTriageEnvClient.from_url(SERVER_URL)
    )

    try:
        for task in tasks:
            await run_episode(env, task, llm)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())