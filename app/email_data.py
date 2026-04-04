"""
app/email_data.py — Email dataset for the Email Triage Environment.

Each entry contains:
  email_id, subject, sender, timestamp, body, thread_history
  ground_truth:
    category        : billing | tech | general | complaint | spam
    priority        : low | medium | high | critical
    action_type     : flag_spam | archive | escalate | respond
    requires_response: True if a written reply is expected
    response_config  : keyword lists used by the response quality grader

Task → email mapping
  email-classify  (easy)   : 3 emails
  email-triage    (medium) : 5 emails
  email-respond   (hard)   : 2 emails
"""

from __future__ import annotations
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Easy task emails  (email-classify — 3 emails)
# ---------------------------------------------------------------------------

_SPAM_001: Dict[str, Any] = {
    "email_id":  "spam-001",
    "subject":   "CONGRATULATIONS! You Have Been Selected to Receive $5,000,000!!!",
    "sender":    "prince.okonkwo@freewebmail.co",
    "timestamp": "2024-01-15T09:23:00Z",
    "thread_history": [],
    "body": (
        "Dear Friend,\n\n"
        "I am Prince Emmanuel Okonkwo, son of the late General Okonkwo of Nigeria. "
        "My father left an unclaimed inheritance of USD $85,000,000. "
        "I need a TRUSTED foreign partner to help transfer this money. "
        "In return you will receive 40% of the total sum.\n\n"
        "Please send your full name, bank account number, and a processing fee of $500.\n\n"
        "May God bless you,\nPrince Emmanuel Okonkwo"
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
        "Hi there,\n\n"
        "I recently purchased a laptop stand from your website (Order #48291, January 10th). "
        "Unfortunately it does not fit my desk setup.\n\n"
        "I would like to return it and get a refund. Could you clarify:\n"
        "1. What is your return window? Am I still eligible at 5 days?\n"
        "2. Do I need to pay return shipping?\n"
        "3. How long does the refund take after you receive the item?\n\n"
        "Thanks,\nSarah Johnson"
    ),
    "ground_truth": {
        "category": "billing",
        "priority": "medium",
        "action_type": "respond",
        "requires_response": True,
        "response_config": {
            "required_keywords": ["return", "refund", "order"],
            "ideal_keywords": ["policy", "window", "shipping", "days", "process"],
            "min_words": 40,
        },
    },
}

_TECH_001: Dict[str, Any] = {
    "email_id":  "tech-001",
    "subject":   "CRITICAL: Production database DOWN — active revenue loss",
    "sender":    "ops-alerts@internal-monitoring.company.com",
    "timestamp": "2024-01-15T11:30:00Z",
    "thread_history": [],
    "body": (
        "AUTOMATED ALERT — SEVERITY: CRITICAL\n\n"
        "System Affected: Primary Production Database (db-prod-01)\n"
        "Status: UNRESPONSIVE\n\n"
        "Impact:\n"
        "  - ALL payment processing is currently OFFLINE\n"
        "  - 63 failed transactions in last 8 minutes\n"
        "  - Estimated loss: $2,340 per minute\n"
        "  - ~1,400 users cannot complete purchases\n\n"
        "IMMEDIATE ESCALATION REQUIRED.\n— Monitoring System"
    ),
    "ground_truth": {
        "category": "tech",
        "priority": "critical",
        "action_type": "escalate",
        "requires_response": False,
    },
}

# ---------------------------------------------------------------------------
# Medium task emails  (email-triage — 5 emails)
# ---------------------------------------------------------------------------

_PHISHING_001: Dict[str, Any] = {
    "email_id":  "phishing-001",
    "subject":   "Urgent: Suspicious login detected — verify your account NOW",
    "sender":    "security-noreply@bankk-secure-alerts.net",
    "timestamp": "2024-01-15T08:15:00Z",
    "thread_history": [],
    "body": (
        "Dear Valued Customer,\n\n"
        "Our fraud detection system flagged UNUSUAL LOGIN ACTIVITY from Romania.\n\n"
        "Click below to verify your identity IMMEDIATELY:\n"
        "  http://secure-bank-verify.auth-check-now.xyz/verify?token=ae82xk\n\n"
        "You must verify within 24 hours or your account will be PERMANENTLY CLOSED.\n\n"
        "Security Department\nFirst National Bank"
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
            "sender": "michael.chen@gmail.com",
            "timestamp": "2024-01-10T09:05:00Z",
            "body": "I ordered a blue wireless mouse (SKU WM-2847-BLU) but received a red wired keyboard.",
        },
        {
            "sender": "support@company.com",
            "timestamp": "2024-01-10T15:30:00Z",
            "body": "Hi Michael, so sorry! We re-shipped the correct item. Expected delivery Jan 13.",
        },
    ],
    "body": (
        "This is unacceptable. I received ANOTHER wrong item — a red keyboard again.\n\n"
        "I ordered: Blue Wireless Mouse SKU WM-2847-BLU ($89.99)\n"
        "I received (TWICE): Red Wired Keyboard SKU KB-1122-RED\n\n"
        "I want:\n"
        "1. A FULL refund of $89.99 TODAY\n"
        "2. Correct product shipped overnight at no cost\n"
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
            "ideal_keywords": ["immediately", "overnight", "correct", "today", "sorry"],
            "min_words": 50,
        },
    },
}

_GENERAL_001: Dict[str, Any] = {
    "email_id":  "general-001",
    "subject":   "Team lunch next Friday — RSVP needed by Wednesday",
    "sender":    "alex.rivera@company.com",
    "timestamp": "2024-01-15T09:00:00Z",
    "thread_history": [],
    "body": (
        "Hey everyone!\n\n"
        "We're organizing a team lunch next Friday (Jan 19) at 12:30 PM "
        "at The Italian Place on Main Street.\n\n"
        "Can you let me know if you can make it by Wednesday EOD?\n\n"
        "Hope to see you all there!\nAlex"
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
    "subject":   "Installation failing with error 0x80070002 — client demo tomorrow",
    "sender":    "raj.patel@startup.io",
    "timestamp": "2024-01-15T16:05:00Z",
    "thread_history": [],
    "body": (
        "Hi Support Team,\n\n"
        "I've been trying to install v3.2.1 on Windows 11 Pro for two hours "
        "and keep hitting:\n\n"
        "  Error Code: 0x80070002\n"
        "  'The system cannot find the file specified'\n\n"
        "Already tried: run as admin, disable antivirus, fresh download, clear %TEMP%.\n"
        "System: Windows 11 Pro 22H2, 16 GB RAM.\n\n"
        "I have a CLIENT DEMO TOMORROW MORNING. Please help urgently.\n\n"
        "Thanks, Raj Patel"
    ),
    "ground_truth": {
        "category": "tech",
        "priority": "high",
        "action_type": "respond",
        "requires_response": True,
        "response_config": {
            "required_keywords": ["install", "error"],
            "ideal_keywords": ["0x80070002", "steps", "windows", "resolve", "contact"],
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
        "Failure will result in:\n"
        "  1. Default judgment against your organization\n"
        "  2. Mandatory award of $47,500 to the claimant\n"
        "  3. Administrative fees of ~$8,200\n\n"
        "Ensure legal counsel has filed all documentation through the arbitration portal.\n\n"
        "Arbitration Services of America"
    ),
    "ground_truth": {
        "category": "complaint",
        "priority": "critical",
        "action_type": "escalate",
        "requires_response": False,
    },
}

# ---------------------------------------------------------------------------
# Hard task emails  (email-respond — 2 emails)
# ---------------------------------------------------------------------------

_COMPLAINT_002: Dict[str, Any] = {
    "email_id":  "complaint-002",
    "subject":   "Enterprise account ENT-28471 terminated without notice — legal action imminent",
    "sender":    "james.morrison@morrisonassociates.com",
    "timestamp": "2024-01-15T16:30:00Z",
    "thread_history": [],
    "body": (
        "Dear Customer Support Manager,\n\n"
        "Our enterprise account (Account ID: ENT-28471) was terminated yesterday "
        "at 3:47 PM without any prior notice.\n\n"
        "Background:\n"
        "  - Morrison & Associates: 4-year premium subscriber\n"
        "  - Monthly spend: $2,400 (87 active users)\n"
        "  - All 87 users currently LOCKED OUT\n"
        "  - Client deliverables due tomorrow morning\n\n"
        "Requiring IMMEDIATE answers:\n"
        "1. WHY was our account terminated with zero warning?\n"
        "2. How do we urgently recover our project data?\n"
        "3. Invoice #INV-2024-0892 ($2,400) was paid — will it be refunded?\n\n"
        "Legal team is briefed. If no senior manager response within 2 HOURS, "
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
                "senior", "manager", "urgently", "resolve",
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
        "We are a certified technology partner with 500+ mutual clients on your payment API v2.3. "
        "We received a deprecation notice setting end-of-life at February 1 — only 17 days away.\n\n"
        "Four critical blockers:\n"
        "1. INCOMPLETE DOCS: Migration guide is missing sections 4.2 and 4.7 "
        "(webhook handling + retry logic).\n"
        "2. INFEASIBLE TIMELINE: 500+ client migrations in 17 days is impossible.\n"
        "3. REVENUE RISK: We process $4M/day — any downtime is catastrophic.\n"
        "4. COMPLIANCE: Two Fortune 500 clients require 60-day advance notice for API changes.\n\n"
        "Formal requests:\n"
        "  a) Extend deprecation deadline by minimum 60 days\n"
        "  b) Publish complete docs for sections 4.2 and 4.7 immediately\n"
        "  c) Assign a dedicated technical migration contact\n"
        "  d) Written confirmation of extension for compliance teams\n\n"
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
                "technical", "contact", "priority", "migration", "partner",
            ],
            "min_words": 80,
        },
    },
}

# ---------------------------------------------------------------------------
# Public dataset maps
# ---------------------------------------------------------------------------

TASK_EMAILS: Dict[str, List[Dict[str, Any]]] = {
    "email-classify": [_SPAM_001, _BILLING_001, _TECH_001],
    "email-triage":   [_PHISHING_001, _BILLING_002, _GENERAL_001, _TECH_002, _COMPLAINT_001],
    "email-respond":  [_COMPLAINT_002, _TECH_003],
}

# Flat lookup by email_id
ALL_EMAILS: Dict[str, Dict[str, Any]] = {
    e["email_id"]: e
    for emails in TASK_EMAILS.values()
    for e in emails
}

# Simplified list — backward-compatible with the OpenEnv build prompt spec
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