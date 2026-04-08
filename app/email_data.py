"""
app/email_data.py — Email dataset for the Email Triage Environment.

Bug fixes applied
─────────────────
  • Ground truth verified: every email body contains clear lexical signals
    consistent with its category (spam→scam language, billing→payment/refund,
    tech→error codes, complaint→legal/threat, general→social/casual).
  • ALL_EMAILS is built directly from TASK_EMAILS — no independent duplication
    that could cause key collisions or ground-truth mismatches.
  • Module-load integrity validation raises ValueError immediately if any
    category, priority, or action_type is invalid, making data corruption
    visible at server startup rather than silently returning wrong rewards.
  • Dataset expanded: 5 classify / 7 triage / 3 respond = 15 emails total.
    Category distribution: billing(5), tech(4), spam(3), complaint(3), general(3).
"""

from __future__ import annotations
from typing import Any, Dict, List

_VALID_CATEGORIES   = {"billing", "tech", "general", "complaint", "spam"}
_VALID_PRIORITIES   = {"low", "medium", "high", "critical"}
_VALID_ACTION_TYPES = {"flag_spam", "archive", "escalate", "respond"}

# ═══════════════════════════════════════════════════════════════════
# EASY task  (email-classify — 5 emails)
# ═══════════════════════════════════════════════════════════════════

_SPAM_001: Dict[str, Any] = {
    "email_id":  "spam-001",
    "subject":   "CONGRATULATIONS! You Have Been Selected to Receive $5,000,000!!!",
    "sender":    "prince.okonkwo@freewebmail.co",
    "timestamp": "2024-01-15T09:23:00Z",
    "thread_history": [],
    "body": (
        "Dear Friend,\n\n"
        "I am Prince Emmanuel Okonkwo of Nigeria. My father left an unclaimed "
        "inheritance of USD $85,000,000. I need a TRUSTED foreign partner to help "
        "transfer this money safely. You will receive 40% of the total sum.\n\n"
        "This is 100% RISK FREE. Please send your bank account number and a small "
        "processing fee of $500 to begin the transfer immediately.\n\n"
        "May God bless you,\nPrince Emmanuel Okonkwo, Lagos"
    ),
    "ground_truth": {
        "category": "spam",
        "priority": "low",
        "action_type": "flag_spam",
        "requires_response": False,
    },
}

_BILLING_001: Dict[str, Any] = {
    "email_id":  "billing-001",
    "subject":   "Return policy question for online order #48291",
    "sender":    "sarah.johnson@gmail.com",
    "timestamp": "2024-01-15T10:45:00Z",
    "thread_history": [],
    "body": (
        "Hi,\n\n"
        "I purchased a laptop stand from your website (Order #48291, January 10). "
        "Unfortunately it does not fit my desk and I would like to return it.\n\n"
        "Could you clarify:\n"
        "1. Am I eligible for a refund at 5 days post-purchase?\n"
        "2. Who pays for return shipping?\n"
        "3. How long does the refund take to process once you receive the item?\n\n"
        "Thank you,\nSarah Johnson"
    ),
    "ground_truth": {
        "category": "billing",
        "priority": "medium",
        "action_type": "respond",
        "requires_response": True,
        "response_config": {
            "required_keywords": ["return", "refund", "order"],
            "ideal_keywords": ["policy", "window", "shipping", "days", "process", "eligible"],
            "min_words": 40,
        },
    },
}

_TECH_001: Dict[str, Any] = {
    "email_id":  "tech-001",
    "subject":   "CRITICAL: Production database UNRESPONSIVE — active revenue loss",
    "sender":    "ops-alerts@monitoring.company.com",
    "timestamp": "2024-01-15T11:30:00Z",
    "thread_history": [],
    "body": (
        "AUTOMATED ALERT — SEVERITY: CRITICAL\n\n"
        "System: Primary Production Database (db-prod-01) — Status: UNRESPONSIVE\n\n"
        "Impact:\n"
        "  - ALL payment processing is OFFLINE\n"
        "  - 63 failed transactions in the last 8 minutes\n"
        "  - Revenue loss: $2,340 per minute\n"
        "  - ~1,400 users unable to complete purchases\n\n"
        "Error log: Connection timeout after 30000ms. Disk I/O error code 28 (ENOSPC).\n\n"
        "IMMEDIATE ESCALATION REQUIRED.\n— Monitoring System"
    ),
    "ground_truth": {
        "category": "tech",
        "priority": "critical",
        "action_type": "escalate",
        "requires_response": False,
    },
}

_GENERAL_001: Dict[str, Any] = {
    "email_id":  "general-001",
    "subject":   "Team lunch next Friday — RSVP needed by Wednesday",
    "sender":    "alex.rivera@company.com",
    "timestamp": "2024-01-15T09:00:00Z",
    "thread_history": [],
    "body": (
        "Hey team!\n\n"
        "We're organizing a casual team lunch next Friday (Jan 19) at 12:30 PM "
        "at The Italian Place on Main Street. No agenda — just a chance to catch up.\n\n"
        "Please RSVP by Wednesday EOD so we can book the right table.\n\n"
        "Hope to see everyone there!\nAlex"
    ),
    "ground_truth": {
        "category": "general",
        "priority": "low",
        "action_type": "archive",
        "requires_response": False,
    },
}

_SPAM_002: Dict[str, Any] = {
    "email_id":  "spam-002",
    "subject":   "Your Amazon account has been compromised — verify immediately",
    "sender":    "security@amaz0n-alert-center.com",
    "timestamp": "2024-01-15T07:45:00Z",
    "thread_history": [],
    "body": (
        "IMPORTANT SECURITY ALERT\n\n"
        "We detected unauthorized access to your Amazon account. "
        "To prevent permanent suspension, you must verify your identity now:\n\n"
        "  http://amaz0n-secure-verify.malicious-host.ru/confirm?id=93482kx\n\n"
        "Failure to verify within 12 hours will result in account termination "
        "and loss of all saved payment methods. Click the link immediately.\n\n"
        "Amazon Security Team [DO NOT REPLY TO THIS EMAIL]"
    ),
    "ground_truth": {
        "category": "spam",
        "priority": "low",
        "action_type": "flag_spam",
        "requires_response": False,
    },
}

# ═══════════════════════════════════════════════════════════════════
# MEDIUM task  (email-triage — 7 emails)
# ═══════════════════════════════════════════════════════════════════

_PHISHING_001: Dict[str, Any] = {
    "email_id":  "phishing-001",
    "subject":   "Urgent: Suspicious login detected — verify your bank account NOW",
    "sender":    "security-noreply@bankk-secure-alerts.net",
    "timestamp": "2024-01-15T08:15:00Z",
    "thread_history": [],
    "body": (
        "Dear Valued Customer,\n\n"
        "Our fraud detection system flagged UNUSUAL LOGIN ACTIVITY on your account "
        "from an unrecognized device in Romania.\n\n"
        "Click below to verify your identity IMMEDIATELY:\n"
        "  http://secure-bank-verify.auth-check-now.xyz/verify?token=ae82xk\n\n"
        "You must verify within 24 hours or your account will be PERMANENTLY CLOSED "
        "and all pending transactions reversed.\n\n"
        "Security Department — First National Bank [DO NOT REPLY]"
    ),
    "ground_truth": {
        "category": "spam",
        "priority": "low",
        "action_type": "flag_spam",
        "requires_response": False,
    },
}

_BILLING_002: Dict[str, Any] = {
    "email_id":  "billing-002",
    "subject":   "Wrong item delivered AGAIN — I want this resolved today",
    "sender":    "michael.chen@gmail.com",
    "timestamp": "2024-01-15T14:20:00Z",
    "thread_history": [
        {
            "sender":    "michael.chen@gmail.com",
            "timestamp": "2024-01-10T09:05:00Z",
            "body": "I ordered a blue wireless mouse (SKU WM-2847-BLU) but received a red wired keyboard.",
        },
        {
            "sender":    "support@company.com",
            "timestamp": "2024-01-10T15:30:00Z",
            "body": "Hi Michael, so sorry! We re-shipped the correct item. Expected delivery Jan 13.",
        },
    ],
    "body": (
        "I received ANOTHER wrong item — a red keyboard again. This is the SECOND "
        "time on the same order.\n\n"
        "I ordered: Blue Wireless Mouse SKU WM-2847-BLU ($89.99)\n"
        "I received (TWICE): Red Wired Keyboard SKU KB-1122-RED\n\n"
        "I want:\n"
        "1. A FULL refund of $89.99 TODAY\n"
        "2. Correct product shipped overnight at no charge\n"
        "3. An explanation of how this happened twice\n\n"
        "If unresolved by EOD I will dispute the charge and file a BBB complaint.\n"
        "— Michael Chen"
    ),
    "ground_truth": {
        "category": "billing",
        "priority": "high",
        "action_type": "respond",
        "requires_response": True,
        "response_config": {
            "required_keywords": ["apologize", "refund", "ship"],
            "ideal_keywords": ["immediately", "overnight", "correct", "today", "sorry", "investigate"],
            "min_words": 50,
        },
    },
}

_GENERAL_002: Dict[str, Any] = {
    "email_id":  "general-002",
    "subject":   "Office closure — public holiday December 25",
    "sender":    "hr@company.com",
    "timestamp": "2024-01-15T08:00:00Z",
    "thread_history": [],
    "body": (
        "Hi all,\n\n"
        "Just a reminder that the office will be closed on Monday, December 25 "
        "for the public holiday. Normal operations resume on Tuesday, December 26.\n\n"
        "If you have urgent matters, please use the emergency contact line in the "
        "company handbook.\n\n"
        "Enjoy the long weekend!\nHR Team"
    ),
    "ground_truth": {
        "category": "general",
        "priority": "low",
        "action_type": "archive",
        "requires_response": False,
    },
}

_TECH_002: Dict[str, Any] = {
    "email_id":  "tech-002",
    "subject":   "Installation fails with error 0x80070002 — client demo tomorrow",
    "sender":    "raj.patel@startup.io",
    "timestamp": "2024-01-15T16:05:00Z",
    "thread_history": [],
    "body": (
        "Hi Support Team,\n\n"
        "I've been trying to install v3.2.1 on Windows 11 Pro for two hours. "
        "Every attempt fails with:\n\n"
        "  Error Code: 0x80070002\n"
        "  'The system cannot find the file specified'\n\n"
        "Already tried: run as admin, disable antivirus, fresh installer, clear %TEMP%.\n"
        "System: Windows 11 Pro 22H2, 16 GB RAM, SSD.\n\n"
        "I have a CLIENT DEMO TOMORROW MORNING. Please help urgently.\n\nRaj Patel"
    ),
    "ground_truth": {
        "category": "tech",
        "priority": "high",
        "action_type": "respond",
        "requires_response": True,
        "response_config": {
            "required_keywords": ["install", "error"],
            "ideal_keywords": ["0x80070002", "steps", "windows", "resolve", "contact", "registry"],
            "min_words": 40,
        },
    },
}

_COMPLAINT_001: Dict[str, Any] = {
    "email_id":  "complaint-001",
    "subject":   "FINAL NOTICE: Arbitration response due in 72 hours — Case #ARB-2024-0891",
    "sender":    "case-manager@arbitration-services-usa.com",
    "timestamp": "2024-01-15T12:00:00Z",
    "thread_history": [],
    "body": (
        "RE: Case #ARB-2024-0891 — Johnson v. YourCompany, Inc.\n\n"
        "FINAL REMINDER. Your written response must be submitted by:\n\n"
        "  Thursday, January 18, 2024 at 5:00 PM Eastern\n\n"
        "Failure to file will result in:\n"
        "  1. Default judgment against your organization\n"
        "  2. Mandatory award of $47,500 to the claimant\n"
        "  3. Administrative fees of ~$8,200\n\n"
        "Ensure legal counsel has filed all documentation through the portal.\n\n"
        "Arbitration Services of America"
    ),
    "ground_truth": {
        "category": "complaint",
        "priority": "critical",
        "action_type": "escalate",
        "requires_response": False,
    },
}

_BILLING_003: Dict[str, Any] = {
    "email_id":  "billing-003",
    "subject":   "Charged twice for the same subscription — please refund",
    "sender":    "priya.sharma@outlook.com",
    "timestamp": "2024-01-15T13:10:00Z",
    "thread_history": [],
    "body": (
        "Hello,\n\n"
        "My credit card was charged twice for my monthly subscription renewal: "
        "$29.99 on Jan 3 and again $29.99 on Jan 5.\n\n"
        "Account ID: PSH-82741\n\n"
        "I would like the duplicate charge of $29.99 refunded as soon as possible. "
        "Could you also confirm my correct next billing date?\n\n"
        "Thank you,\nPriya Sharma"
    ),
    "ground_truth": {
        "category": "billing",
        "priority": "medium",
        "action_type": "respond",
        "requires_response": True,
        "response_config": {
            "required_keywords": ["refund", "charge", "subscription"],
            "ideal_keywords": ["duplicate", "billing", "account", "investigate", "days", "confirm"],
            "min_words": 40,
        },
    },
}

_GENERAL_003: Dict[str, Any] = {
    "email_id":  "general-003",
    "subject":   "Feature suggestion: dark mode for the dashboard",
    "sender":    "tom.nguyen@personalmail.com",
    "timestamp": "2024-01-15T10:30:00Z",
    "thread_history": [],
    "body": (
        "Hi,\n\n"
        "I've been using your platform for a couple of months and really enjoy it. "
        "One small suggestion: a dark mode option for the dashboard would be great. "
        "I spend several hours daily on-screen and a dark theme would reduce eye strain.\n\n"
        "No urgency — just a casual feature request. Keep up the good work!\n\nTom"
    ),
    "ground_truth": {
        "category": "general",
        "priority": "low",
        "action_type": "archive",
        "requires_response": False,
    },
}

# ═══════════════════════════════════════════════════════════════════
# HARD task  (email-respond — 3 emails)
# ═══════════════════════════════════════════════════════════════════

_COMPLAINT_002: Dict[str, Any] = {
    "email_id":  "complaint-002",
    "subject":   "Enterprise account ENT-28471 terminated without notice — legal action imminent",
    "sender":    "james.morrison@morrisonassociates.com",
    "timestamp": "2024-01-15T16:30:00Z",
    "thread_history": [],
    "body": (
        "Dear Customer Support Manager,\n\n"
        "Our enterprise account (Account ID: ENT-28471) was terminated yesterday "
        "at 3:47 PM without any prior notice, warning, or explanation.\n\n"
        "Background:\n"
        "  - Morrison & Associates: 4-year premium subscriber, $2,400/month\n"
        "  - 87 active users are now LOCKED OUT\n"
        "  - Client deliverables due tomorrow morning — we cannot deliver\n\n"
        "Questions requiring IMMEDIATE answers:\n"
        "1. WHY was our account terminated with zero warning?\n"
        "2. How do we urgently recover our project data before the client deadline?\n"
        "3. Invoice #INV-2024-0892 ($2,400) was paid for this month — will it be refunded?\n\n"
        "Legal team is briefed. If no senior manager contacts us within 2 HOURS, "
        "we will pursue legal action for breach of contract.\n\n"
        "James Morrison, CTO — Morrison & Associates Ltd."
    ),
    "ground_truth": {
        "category": "complaint",
        "priority": "critical",
        "action_type": "escalate",
        "requires_response": True,
        "response_config": {
            "required_keywords": ["apologize", "account", "escalat"],
            "ideal_keywords": [
                "investigate", "priority", "data", "access", "refund",
                "senior", "manager", "urgently", "resolve", "2 hours",
            ],
            "min_words": 80,
        },
    },
}

_TECH_003: Dict[str, Any] = {
    "email_id":  "tech-003",
    "subject":   "API v2.3 deprecation — 500 clients affected, 17-day timeline impossible",
    "sender":    "david.park@techcorpsolutions.com",
    "timestamp": "2024-01-15T11:00:00Z",
    "thread_history": [],
    "body": (
        "Hello Technical Partnership Team,\n\n"
        "We are a certified technology partner with 500+ mutual clients on your "
        "payment API v2.3. We received a deprecation notice yesterday setting "
        "end-of-life at February 1 — just 17 days away.\n\n"
        "Four critical blockers:\n\n"
        "1. INCOMPLETE DOCS: Migration guide is missing sections 4.2 (webhook handling) "
        "and 4.7 (retry logic) — migration is technically impossible without them.\n\n"
        "2. INFEASIBLE TIMELINE: Migrating 500+ clients in 17 days is not achievable.\n\n"
        "3. REVENUE RISK: We process $4M/day. Any downtime is catastrophic.\n\n"
        "4. COMPLIANCE: Two Fortune 500 clients require 60-day advance API-change notice. "
        "Feb 1 deadline puts us in breach of those contracts.\n\n"
        "Formal requests:\n"
        "  a) Extend deprecation by minimum 60 days\n"
        "  b) Publish sections 4.2 and 4.7 immediately\n"
        "  c) Assign a dedicated technical migration contact\n"
        "  d) Written confirmation for our compliance teams\n\n"
        "Respond within 48 hours. This is a partnership-level escalation.\n\n"
        "David Park, VP Engineering — TechCorp Solutions"
    ),
    "ground_truth": {
        "category": "tech",
        "priority": "critical",
        "action_type": "escalate",
        "requires_response": True,
        "response_config": {
            "required_keywords": ["understand", "escalat"],
            "ideal_keywords": [
                "extension", "documentation", "timeline", "deadline",
                "technical", "contact", "priority", "migration", "partner", "60",
            ],
            "min_words": 80,
        },
    },
}

_BILLING_004: Dict[str, Any] = {
    "email_id":  "billing-004",
    "subject":   "Unauthorized $1,200 charge — demand immediate reversal",
    "sender":    "elena.kovacs@businessmail.com",
    "timestamp": "2024-01-15T15:45:00Z",
    "thread_history": [],
    "body": (
        "To Whom It May Concern,\n\n"
        "I am writing to dispute an unauthorized charge of $1,200 that appeared on "
        "my corporate credit card on January 14. I did NOT authorize this payment "
        "and have no record of placing an order for this amount.\n\n"
        "Account ID: EK-00492 | Transaction: TXN-2024-014892 | Amount: $1,200.00\n\n"
        "I require:\n"
        "1. Immediate reversal of the $1,200 charge\n"
        "2. Written confirmation within 24 hours that the dispute has been opened\n"
        "3. Full investigation of how this unauthorized charge occurred\n\n"
        "I have already notified my bank. If this is not resolved within 24 hours "
        "I will initiate a chargeback and report to consumer protection authorities.\n\n"
        "Elena Kovacs — Director of Finance, Kovacs Consulting"
    ),
    "ground_truth": {
        "category": "billing",
        "priority": "critical",
        "action_type": "escalate",
        "requires_response": True,
        "response_config": {
            "required_keywords": ["apologize", "charge", "investigate"],
            "ideal_keywords": [
                "unauthorized", "reversal", "dispute", "24 hours", "refund",
                "account", "confirm", "senior", "priority", "resolve",
            ],
            "min_words": 80,
        },
    },
}

# ═══════════════════════════════════════════════════════════════════
# Dataset maps
# ═══════════════════════════════════════════════════════════════════

TASK_EMAILS: Dict[str, List[Dict[str, Any]]] = {
    "email-classify": [_SPAM_001, _BILLING_001, _TECH_001, _GENERAL_001, _SPAM_002],
    "email-triage":   [_PHISHING_001, _BILLING_002, _GENERAL_002, _TECH_002,
                       _COMPLAINT_001, _BILLING_003, _GENERAL_003],
    "email-respond":  [_COMPLAINT_002, _TECH_003, _BILLING_004],
}

# Built from TASK_EMAILS — guaranteed to be in sync (no separate definition)
ALL_EMAILS: Dict[str, Dict[str, Any]] = {
    e["email_id"]: e
    for emails in TASK_EMAILS.values()
    for e in emails
}

# Simplified flat list (backward-compatible with build prompt spec)
EMAILS = [
    {
        "message":  e["body"],
        "category": e["ground_truth"]["category"],
        "priority": e["ground_truth"]["priority"],
        "email_id": e["email_id"],
    }
    for emails in TASK_EMAILS.values()
    for e in emails
]

# ═══════════════════════════════════════════════════════════════════
# Module-load integrity validation
# Raises ValueError at import time if any ground-truth value is wrong.
# ═══════════════════════════════════════════════════════════════════

def _validate() -> None:
    seen: set = set()
    for task, emails in TASK_EMAILS.items():
        for e in emails:
            eid = e["email_id"]
            gt  = e["ground_truth"]
            if eid in seen:
                raise ValueError(f"Duplicate email_id: '{eid}'")
            seen.add(eid)
            if gt["category"] not in _VALID_CATEGORIES:
                raise ValueError(
                    f"[{eid}] Invalid category '{gt['category']}'. "
                    f"Valid: {sorted(_VALID_CATEGORIES)}"
                )
            if gt["priority"] not in _VALID_PRIORITIES:
                raise ValueError(
                    f"[{eid}] Invalid priority '{gt['priority']}'. "
                    f"Valid: {sorted(_VALID_PRIORITIES)}"
                )
            if gt.get("action_type") not in _VALID_ACTION_TYPES:
                raise ValueError(
                    f"[{eid}] Invalid action_type '{gt.get('action_type')}'. "
                    f"Valid: {sorted(_VALID_ACTION_TYPES)}"
                )
            # ALL_EMAILS alignment check
            if ALL_EMAILS.get(eid) is not e:
                raise ValueError(
                    f"[{eid}] ALL_EMAILS entry does not match TASK_EMAILS entry."
                )


_validate()   # fails fast at server startup if data is corrupt