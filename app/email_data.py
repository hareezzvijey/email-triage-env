"""
app/email_data.py — Email dataset for the Email Triage Environment.
"""

from __future__ import annotations
from typing import Any, Dict, List

_VALID_CATEGORIES = {"billing", "tech", "general", "complaint", "spam"}
_VALID_PRIORITIES = {"low", "medium", "high", "critical"}
_VALID_ACTION_TYPES = {"flag_spam", "archive", "escalate", "respond"}

# ============================================================
# EASY task (email-classify — 5 emails)
# ============================================================

_SPAM_001: Dict[str, Any] = {
    "email_id": "spam-001",
    "subject": "CONGRATULATIONS! You Have Been Selected to Receive $5,000,000!!!",
    "sender": "prince.okonkwo@freewebmail.co",
    "timestamp": "2024-01-15T09:23:00Z",
    "thread_history": [],
    "body": "Dear Friend,\n\nI am Prince Emmanuel Okonkwo of Nigeria. My father left an unclaimed inheritance of USD $85,000,000. I need a TRUSTED foreign partner to help transfer this money safely. You will receive 40% of the total sum.\n\nPlease send your bank account number and a processing fee of $500.\n\nMay God bless you,\nPrince Emmanuel Okonkwo",
    "ground_truth": {"category": "spam", "priority": "low", "action_type": "flag_spam", "requires_response": False},
}

_BILLING_001: Dict[str, Any] = {
    "email_id": "billing-001",
    "subject": "Return policy question for online order #48291",
    "sender": "sarah.johnson@gmail.com",
    "timestamp": "2024-01-15T10:45:00Z",
    "thread_history": [],
    "body": "Hi,\n\nI purchased a laptop stand from your website (Order #48291, January 10). Unfortunately it does not fit my desk and I would like to return it.\n\nCould you clarify:\n1. Am I eligible for a refund at 5 days post-purchase?\n2. Who pays for return shipping?\n3. How long does the refund take to process?\n\nThank you,\nSarah Johnson",
    "ground_truth": {"category": "billing", "priority": "medium", "action_type": "respond", "requires_response": True, "response_config": {"required_keywords": ["return", "refund", "order"], "ideal_keywords": ["policy", "shipping", "days"], "min_words": 40}},
}

_TECH_001: Dict[str, Any] = {
    "email_id": "tech-001",
    "subject": "CRITICAL: Production database UNRESPONSIVE — active revenue loss",
    "sender": "ops-alerts@monitoring.company.com",
    "timestamp": "2024-01-15T11:30:00Z",
    "thread_history": [],
    "body": "AUTOMATED ALERT — SEVERITY: CRITICAL\n\nSystem: Primary Production Database (db-prod-01) — Status: UNRESPONSIVE\n\nImpact:\n  - ALL payment processing is OFFLINE\n  - Revenue loss: $2,340 per minute\n\nIMMEDIATE ESCALATION REQUIRED.",
    "ground_truth": {"category": "tech", "priority": "critical", "action_type": "escalate", "requires_response": False},
}

_GENERAL_001: Dict[str, Any] = {
    "email_id": "general-001",
    "subject": "Team lunch next Friday — RSVP needed by Wednesday",
    "sender": "alex.rivera@company.com",
    "timestamp": "2024-01-15T09:00:00Z",
    "thread_history": [],
    "body": "Hey team!\n\nWe're organizing a casual team lunch next Friday at 12:30 PM at The Italian Place.\n\nPlease RSVP by Wednesday.\n\nAlex",
    "ground_truth": {"category": "general", "priority": "low", "action_type": "archive", "requires_response": False},
}

_SPAM_002: Dict[str, Any] = {
    "email_id": "spam-002",
    "subject": "Your Amazon account has been compromised — verify immediately",
    "sender": "security@amaz0n-alert-center.com",
    "timestamp": "2024-01-15T07:45:00Z",
    "thread_history": [],
    "body": "IMPORTANT SECURITY ALERT\n\nWe detected unauthorized access to your Amazon account.\n\nhttp://amaz0n-secure-verify.malicious-host.ru/confirm\n\nAmazon Security Team",
    "ground_truth": {"category": "spam", "priority": "low", "action_type": "flag_spam", "requires_response": False},
}

# ============================================================
# MEDIUM task (email-triage — 7 emails)
# ============================================================

_PHISHING_001: Dict[str, Any] = {
    "email_id": "phishing-001",
    "subject": "Urgent: Suspicious login detected — verify your bank account NOW",
    "sender": "security-noreply@bankk-secure-alerts.net",
    "timestamp": "2024-01-15T08:15:00Z",
    "thread_history": [],
    "body": "Dear Valued Customer,\n\nOur fraud detection system flagged UNUSUAL LOGIN ACTIVITY on your account from Romania.\n\nClick below to verify your identity IMMEDIATELY:\n  http://secure-bank-verify.auth-check-now.xyz/verify\n\nYou must verify within 24 hours or your account will be PERMANENTLY CLOSED.",
    "ground_truth": {"category": "spam", "priority": "low", "action_type": "flag_spam", "requires_response": False},
}

_BILLING_002: Dict[str, Any] = {
    "email_id": "billing-002",
    "subject": "Wrong item delivered AGAIN — I want this resolved today",
    "sender": "michael.chen@gmail.com",
    "timestamp": "2024-01-15T14:20:00Z",
    "thread_history": [],
    "body": "I received ANOTHER wrong item. I ordered a Blue Wireless Mouse but received a Red Wired Keyboard twice.\n\nI want a FULL refund and the correct product shipped overnight.\n\nIf unresolved by EOD I will dispute the charge.",
    "ground_truth": {"category": "billing", "priority": "high", "action_type": "respond", "requires_response": True, "response_config": {"required_keywords": ["apologize", "refund", "ship"], "ideal_keywords": ["overnight", "correct", "sorry"], "min_words": 50}},
}

_GENERAL_002: Dict[str, Any] = {
    "email_id": "general-002",
    "subject": "Office closure — public holiday December 25",
    "sender": "hr@company.com",
    "timestamp": "2024-01-15T08:00:00Z",
    "thread_history": [],
    "body": "Hi all,\n\nJust a reminder that the office will be closed on Monday, December 25 for the public holiday.\n\nEnjoy the long weekend!\nHR Team",
    "ground_truth": {"category": "general", "priority": "low", "action_type": "archive", "requires_response": False},
}

_TECH_002: Dict[str, Any] = {
    "email_id": "tech-002",
    "subject": "Installation fails with error 0x80070002 — client demo tomorrow",
    "sender": "raj.patel@startup.io",
    "timestamp": "2024-01-15T16:05:00Z",
    "thread_history": [],
    "body": "Hi Support Team,\n\nInstallation fails with Error Code: 0x80070002 'The system cannot find the file specified'.\n\nI have a CLIENT DEMO TOMORROW MORNING. Please help urgently.",
    "ground_truth": {"category": "tech", "priority": "high", "action_type": "respond", "requires_response": True, "response_config": {"required_keywords": ["install", "error"], "ideal_keywords": ["0x80070002", "steps", "resolve"], "min_words": 40}},
}

_COMPLAINT_001: Dict[str, Any] = {
    "email_id": "complaint-001",
    "subject": "FINAL NOTICE: Arbitration response due in 72 hours — Case #ARB-2024-0891",
    "sender": "case-manager@arbitration-services-usa.com",
    "timestamp": "2024-01-15T12:00:00Z",
    "thread_history": [],
    "body": "RE: Case #ARB-2024-0891 — Johnson v. YourCompany, Inc.\n\nFINAL REMINDER. Your written response must be submitted by Thursday, January 18, 2024.\n\nFailure will result in default judgment of $47,500.",
    "ground_truth": {"category": "complaint", "priority": "critical", "action_type": "escalate", "requires_response": False},
}

_BILLING_003: Dict[str, Any] = {
    "email_id": "billing-003",
    "subject": "Charged twice for the same subscription — please refund",
    "sender": "priya.sharma@outlook.com",
    "timestamp": "2024-01-15T13:10:00Z",
    "thread_history": [],
    "body": "Hello,\n\nMy credit card was charged twice for my monthly subscription: $29.99 on Jan 3 and again $29.99 on Jan 5.\n\nPlease refund the duplicate charge.",
    "ground_truth": {"category": "billing", "priority": "medium", "action_type": "respond", "requires_response": True, "response_config": {"required_keywords": ["refund", "charge", "subscription"], "ideal_keywords": ["duplicate", "billing", "investigate"], "min_words": 40}},
}

_GENERAL_003: Dict[str, Any] = {
    "email_id": "general-003",
    "subject": "Feature suggestion: dark mode for the dashboard",
    "sender": "tom.nguyen@personalmail.com",
    "timestamp": "2024-01-15T10:30:00Z",
    "thread_history": [],
    "body": "Hi,\n\nA small suggestion: a dark mode option for the dashboard would be great. I spend hours on-screen and dark theme would reduce eye strain.\n\nNo urgency — just a casual feature request.",
    "ground_truth": {"category": "general", "priority": "low", "action_type": "archive", "requires_response": False},
}

# ============================================================
# HARD task (email-respond — 3 emails)
# ============================================================

_COMPLAINT_002: Dict[str, Any] = {
    "email_id": "complaint-002",
    "subject": "Enterprise account ENT-28471 terminated without notice — legal action imminent",
    "sender": "james.morrison@morrisonassociates.com",
    "timestamp": "2024-01-15T16:30:00Z",
    "thread_history": [],
    "body": "Dear Customer Support Manager,\n\nOur enterprise account was terminated without notice. 87 active users are LOCKED OUT. Client deliverables due tomorrow.\n\nLegal team is briefed. If no senior manager contacts us within 2 HOURS, we will pursue legal action.",
    "ground_truth": {"category": "complaint", "priority": "critical", "action_type": "escalate", "requires_response": True, "response_config": {"required_keywords": ["apologize", "account", "escalat"], "ideal_keywords": ["investigate", "priority", "access", "refund", "senior"], "min_words": 80}},
}

_TECH_003: Dict[str, Any] = {
    "email_id": "tech-003",
    "subject": "API v2.3 deprecation — 500 clients affected, 17-day timeline impossible",
    "sender": "david.park@techcorpsolutions.com",
    "timestamp": "2024-01-15T11:00:00Z",
    "thread_history": [],
    "body": "Hello Technical Partnership Team,\n\nAPI v2.3 deprecation set for Feb 1 — only 17 days away. We process $4M/day. Any downtime is catastrophic.\n\nRequesting 60-day extension and dedicated migration contact.",
    "ground_truth": {"category": "tech", "priority": "critical", "action_type": "escalate", "requires_response": True, "response_config": {"required_keywords": ["understand", "escalat"], "ideal_keywords": ["extension", "documentation", "timeline", "deadline", "technical"], "min_words": 80}},
}

_BILLING_004: Dict[str, Any] = {
    "email_id": "billing-004",
    "subject": "Unauthorized $1,200 charge — demand immediate reversal",
    "sender": "elena.kovacs@businessmail.com",
    "timestamp": "2024-01-15T15:45:00Z",
    "thread_history": [],
    "body": "To Whom It May Concern,\n\nUnauthorized charge of $1,200 appeared on my corporate credit card. I did NOT authorize this.\n\nI require immediate reversal and written confirmation.",
    "ground_truth": {"category": "billing", "priority": "critical", "action_type": "escalate", "requires_response": True, "response_config": {"required_keywords": ["apologize", "charge", "investigate"], "ideal_keywords": ["unauthorized", "reversal", "dispute", "24 hours"], "min_words": 80}},
}

# ============================================================
# Dataset maps
# ============================================================

TASK_EMAILS: Dict[str, List[Dict[str, Any]]] = {
    "email-classify": [_SPAM_001, _BILLING_001, _TECH_001, _GENERAL_001, _SPAM_002],
    "email-triage": [_PHISHING_001, _BILLING_002, _GENERAL_002, _TECH_002, _COMPLAINT_001, _BILLING_003, _GENERAL_003],
    "email-respond": [_COMPLAINT_002, _TECH_003, _BILLING_004],
}

ALL_EMAILS: Dict[str, Dict[str, Any]] = {
    e["email_id"]: e for emails in TASK_EMAILS.values() for e in emails
}