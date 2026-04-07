---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# 📬 Email Triage Environment

> **OpenEnv-compliant real-world benchmark for AI email triage and customer-support response generation.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://github.com/openenv)
[![HF Spaces](https://img.shields.io/badge/🤗-HF%20Spaces-yellow)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

---

## 🌍 Motivation

Email triage is one of the most cognitively demanding real-world tasks: agents must simultaneously classify intent, assess urgency, choose a routing action, and draft a professional response — all under time pressure. This environment captures that full complexity in a containerised, reproducible benchmark with three difficulty tiers.

**Total Emails:** 15 (5 classify, 7 triage, 3 respond)

---

## 📁 Project Structure
email-triage-env/

├── server/

│ ├── app.py # FastAPI server + all required endpoints

│ └── environment.py # EmailEnv class — reset() / step() / state()

├── app/

│ ├── models.py # Typed Pydantic models (OpenEnv-compliant)

│ ├── rewards.py # Deterministic graders — zero LLM calls

│ ├── email_data.py # 15 realistic email scenarios with ground truth

│ └── client.py # Async HTTP client (extends EnvClient)

├── inference.py # Baseline LLM agent — mandatory stdout format

├── openenv.yaml # OpenEnv metadata and API spec

├── Dockerfile # HF Spaces-compatible container

├── requirements.txt

└── README.md

---

## 🤖 Core RL Loop

```python
obs = env.reset()
while not done:
    action = agent(obs)
    obs, reward, done, _ = env.step(action)

🏗️ Architecture
Agent (LLM)
    │  EmailAction(category, priority, action_type, response)
    ▼
inference.py  →  app/client.py  →  POST /step
                                        │
                               server/app.py (FastAPI)
                               /reset /step /state /tasks /grader /baseline
                                        │
                               server/environment.py (EmailEnv)
                               reset() → step() → state()
                                        │
                               app/rewards.py
                               compute_reward()  [deterministic]
```
📋 Tasks
Task 1 — email-classify · Easy · 5 emails
Field	Weight	Detail
category	40%	billing / tech / general / complaint / spam
priority	30%	exact=1.0 · adjacent rank=0.5 · wrong=0.0
reasoning	20%	any non-empty justification
response bonus	10%	optional — small bonus
Emails: Nigerian-prince spam · billing return query · critical DB outage alert · team lunch · Amazon phishing

Task 2 — email-triage · Medium · 7 emails
Field	Weight	Detail
category	20%	
priority	15%	
action_type	50%	flag_spam / archive / escalate / respond
reasoning	10%	
response bonus	5%	
False escalation	−30%	Escalating spam or general emails
Emails: phishing attempt · repeat wrong-item complaint · office closure · install error · legal arbitration deadline · double charge · feature request

Task 3 — email-respond · Hard · 3 emails
Field	Weight
category	10%
priority	10%
action_type	10%
response quality	65%
reasoning	5%
Response quality sub-scores:

Sub-signal	Weight
Required keywords	20%
Ideal keyword coverage	15%
Professional structure (greeting + closing)	10%
Acknowledgment / empathy	10%
Length ≥ 80 words	10%
Emails: enterprise account terminated without notice · API deprecation crisis affecting 500 clients + $4M/day revenue · unauthorized $1,200 charge

```
🔭 Observation Space
class EmailObservation(Observation):
    message:          str        # Full email body (OpenEnv spec field)
    history:          list[str]  # Thread history as formatted strings
    subject:          str
    sender:           str
    timestamp:        str        # ISO-8601
    email_id:         str
    task:             str
    task_description: str        # Instructions + scoring rubric
    step:             int
    emails_remaining: int
    total_emails:     int
    inbox_size:       int        # Alias for emails_remaining
```
⚡ Action Space
class EmailAction(Action):
    category:    str            # billing|tech|general|complaint|spam (required)
    priority:    str            # low|medium|high|critical (required)
    action_type: str | None     # classify|flag_spam|archive|escalate|respond
    response:    str | None     # Full response draft (hard task — graded)
    reasoning:   str | None     # Justification (bonus reward)
```
🚀 Quick Start
Docker
git clone https://github.com/hareezzvijey/email-triage-env
cd email-triage-env

docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

curl http://localhost:7860/health
# → {"status":"ok","active_sessions":0,...}

Local Python
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```
🤖 Running the Baseline Agent
export HF_TOKEN="hf_..."
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama3-8b-8192"
export EMAIL_TRIAGE_SERVER_URL="http://localhost:7860"

# All 3 tasks
python inference.py

# Single task
EMAIL_TRIAGE_TASK=email-respond python inference.py

Expected stdout:
[START] task=email-classify env=email-triage-env model=llama3-8b-8192
[STEP] step=1 action=(cat=spam,pri=low,act=classify) reward=0.90 done=false error=null
[STEP] step=2 action=(cat=billing,pri=medium,act=classify) reward=1.00 done=false error=null
[STEP] step=3 action=(cat=tech,pri=critical,act=classify) reward=0.90 done=false error=null
[STEP] step=4 action=(cat=general,pri=low,act=classify) reward=0.90 done=false error=null
[STEP] step=5 action=(cat=spam,pri=low,act=classify) reward=0.90 done=true error=null
[END] success=true steps=5 rewards=0.90,1.00,0.90,0.90,0.90
```
🐍 Python Client
import asyncio
from app.client import EmailTriageEnvClient
from app.models import EmailAction

async def main():
    async with await EmailTriageEnvClient.from_url("https://hareezz-email-triage-env.hf.space") as env:
        obs = await env.reset(task="email-triage")
        done = False
        while not done:
            action = EmailAction(
                category="billing", priority="high",
                action_type="respond",
                reasoning="Customer complaint requiring a reply"
            )
            result = await env.step(action)
            obs = result.observation
            done = result.done
            print(f"reward={result.reward:.3f}")

asyncio.run(main())
```
📡 API Reference
Method	Path	Description
GET	/health	Liveness probe
GET	/	Environment info
GET	/tasks	Task list + metadata
GET	/baseline	Pre-computed heuristic baseline scores
POST	/reset	Start new episode — body: {"task":"email-triage"}
POST	/step	Submit action — requires X-Session-ID header
GET	/state	Full episode state — requires X-Session-ID header
POST	/grader	Grade any action without running an episode
GET	/docs	Swagger UI

POST /grader — standalone grading
curl -X POST https://hareezz-email-triage-env.hf.space/grader \
  -H "Content-Type: application/json" \
  -d '{
    "email_id": "billing-001",
    "task": "email-triage",
    "action": {
      "category": "billing",
      "priority": "medium",
      "action_type": "respond",
      "reasoning": "return query"
    }
  }'
```
📊 Baseline Scores (Llama 3 8B)
Task	Mean Reward	Success
email-classify (easy)	~0.90	✅
email-triage (medium)	~0.85	✅
email-respond (hard)	~0.75	✅
Note: Scores vary with model quality. Larger models (GPT-4, Claude, Qwen-72B) achieve 90%+ across all tasks.

All graders are fully deterministic — run python inference.py to reproduce.
```
📄 License
Apache 2.0
```
🏆 Live Space
API Endpoint: https://hareezz-email-triage-env.hf.space
