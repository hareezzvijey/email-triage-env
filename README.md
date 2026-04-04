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

---

## 📁 Project Structure
email-triage-env/
├── server/
│ ├── app.py FastAPI server + all required endpoints
│ └── environment.py EmailEnv class — reset() / step() / state()
│
├── app/
│ ├── models.py Typed Pydantic models (OpenEnv-compliant)
│ ├── rewards.py Deterministic graders — zero LLM calls
│ ├── email_data.py 10 realistic email scenarios with ground truth
│ └── client.py Async HTTP client (extends EnvClient)
│
├── inference.py Baseline LLM agent — mandatory stdout format
├── openenv.yaml OpenEnv metadata and API spec
├── Dockerfile HF Spaces-compatible container
├── requirements.txt
└── README.md

text

---

## 🤖 Core RL Loop

```python
obs = env.reset()
while not done:
    action = agent(obs)
    obs, reward, done, _ = env.step(action)
🏗️ Architecture
text
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
📋 Tasks
Task 1 — email-classify · Easy · 3 emails
Field	Weight	Detail
category	40%	billing / tech / general / complaint / spam
priority	30%	exact=1.0 · adjacent rank=0.5 · wrong=0.0
reasoning	20%	any non-empty justification
response bonus	10%	optional — small bonus
Emails: Nigerian-prince spam · billing return query · critical DB outage alert

Task 2 — email-triage · Medium · 5 emails
Field	Weight	Detail
category	20%	
priority	15%	
action_type	50%	flag_spam / archive / escalate / respond
reasoning	10%	
response bonus	5%	
False escalation	−30%	Escalating spam or general emails
Emails: phishing attempt · repeat wrong-item complaint · team lunch invite · install error · legal arbitration deadline

Task 3 — email-respond · Hard · 2 emails
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
Emails: enterprise account terminated without notice · API deprecation crisis affecting 500 clients + $4M/day revenue

🔭 Observation Space
python
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
⚡ Action Space
python
class EmailAction(Action):
    category:    str            # billing|tech|general|complaint|spam  (required)
    priority:    str            # low|medium|high|critical             (required)
    action_type: str | None     # classify|flag_spam|archive|escalate|respond
    response:    str | None     # Full response draft (hard task — graded)
    reasoning:   str | None     # Justification (bonus reward)
🚀 Quick Start
Docker
bash
git clone https://huggingface.co/spaces/hareezz/email-triage-env
cd email-triage-env

docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

curl http://localhost:7860/health
# → {"status":"ok","active_sessions":0,...}
Local Python
bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
🤖 Running the Baseline Agent
bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export EMAIL_TRIAGE_SERVER_URL="http://localhost:7860"

# All 3 tasks
python inference.py

# Single task
EMAIL_TRIAGE_TASK=email-respond python inference.py

# Against a local Docker image
LOCAL_IMAGE_NAME=email-triage-env python inference.py
Expected stdout:

text
[START] task=email-classify env=email-triage-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=(cat=spam,pri=low,act=classify) reward=0.90 done=false error=null
[STEP] step=2 action=(cat=billing,pri=medium,act=classify) reward=1.00 done=false error=null
[STEP] step=3 action=(cat=tech,pri=critical,act=classify) reward=1.00 done=true error=null
[END] success=true steps=3 rewards=0.90,1.00,1.00
🐍 Python Client
python
import asyncio
from app.client import EmailTriageEnvClient
from app.models import EmailAction

async def main():
    async with await EmailTriageEnvClient.from_url("http://localhost:7860") as env:
        obs  = await env.reset(task="email-triage")
        done = False
        while not done:
            action = EmailAction(
                category="billing", priority="high",
                action_type="respond",
                reasoning="Customer complaint requiring a reply"
            )
            result = await env.step(action)
            obs    = result.observation
            done   = result.done
            print(f"reward={result.reward:.3f}")

asyncio.run(main())
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
bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{
    "email_id": "billing-001",
    "task": "email-triage",
    "action": { "category":"billing","priority":"medium","action_type":"respond","reasoning":"return query" }
  }'
📊 Baseline Scores (Qwen/Qwen2.5-72B-Instruct)
Task	Mean Reward	Success
email-classify (easy)	~0.87	✅
email-triage (medium)	~0.73	✅
email-respond (hard)	~0.56	✅
All graders are fully deterministic — run python inference.py to reproduce.

📄 License
Apache 2.0

text

## 🚀 **How to Update Your Space:**

### **Option 1: Edit directly on HF Website (Easiest)**
1. Go to https://huggingface.co/spaces/hareezz/email-triage-env/blob/main/README.md
2. Click **"Edit"**
3. Replace the entire content with the version above (with metadata at top)
4. Click **"Commit changes"**

### **Option 2: Push via Git**
```powershell
cd C:\Users\haree\Downloads\email-triage-env

# Replace your README.md with the new version
# Then:
git add README.md
git commit -m "Add required YAML metadata for Docker Space"
git push origin master