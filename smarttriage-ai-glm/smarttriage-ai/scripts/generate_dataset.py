"""Generate a SYNTHETIC placeholder tickets.csv (300 rows) for the MVP.

This is NOT the real assignment dataset — none was provided in upload/.
It exists so the MVP is runnable end-to-end. Replace data/tickets.csv with the
real assignment dataset before any evaluation. The schema matches the
assignment: id, received_at, customer, subject, body.
"""
from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import settings

random.seed(42)

CUSTOMERS = [
    "alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi",
    "ivan", "judy", "mallory", "nancy", "oscar", "pat", "ruth", "steve",
    "trent", "victor", "wendy", "zoe",
]

# Templates per category. We deliberately include negations and risk keywords
# (refund, urgent, failed, cannot, never) so the zero-shot model has signal.
TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "Billing": [
        ("Double charged on my credit card", "I was charged twice for the same subscription on Nov 3. Please refund the duplicate charge.", "High"),
        ("Invoice never arrived", "I never received my invoice for October and cannot access it in the portal.", "Medium"),
        ("Refund request - urgent", "I cancelled my plan two weeks ago and still have not received my refund. This is urgent.", "Urgent"),
        ("Overcharged after upgrade", "After upgrading to Pro I was overcharged by $40. I need a refund and an explanation.", "High"),
        ("Payment failed but card charged", "My payment failed at checkout but my card was still charged. Please fix this immediately.", "High"),
        ("Cannot update billing details", "I cannot update my billing details - the form keeps erroring out.", "Medium"),
        ("Subscription renewed without consent", "My subscription renewed automatically even though I cancelled. Refund please.", "High"),
        ("Receipt for tax purposes", "Could you send a receipt for my last payment for tax purposes?", "Low"),
    ],
    "Technical": [
        ("App crashes on login", "The mobile app crashes every time I try to log in. Reinstalling did not help.", "High"),
        ("Export feature broken", "The CSV export button does nothing when clicked. It worked last week.", "Medium"),
        ("Slow dashboard performance", "The dashboard takes over 30 seconds to load every page. This is slowing my whole team.", "Medium"),
        ("500 error on reports page", "I get a 500 error when opening the reports page. Please fix this urgently.", "High"),
        ("Cannot upload files", "File uploads fail silently for anything over 5MB. No error message appears.", "Medium"),
        ("Notifications not arriving", "Email notifications stopped arriving last Monday. Nothing in spam either.", "Medium"),
        ("Search returns wrong results", "The search returns completely unrelated results for common queries.", "Low"),
        ("Service down - urgent", "Our entire workspace is down and we cannot reach any data. This is urgent - production is blocked.", "Urgent"),
    ],
    "Account": [
        ("Cannot log in", "I cannot log in - the password reset email never arrives. I am locked out.", "High"),
        ("Account compromised", "I think my account was compromised - there are logins from a country I never visited. Security breach.", "Urgent"),
        ("Two-factor auth broken", "My 2FA codes are rejected even though the time is correct. I cannot access my account.", "High"),
        ("Change email address", "How do I change the email address linked to my account?", "Low"),
        ("Delete my account", "Please delete my account and all associated data immediately.", "Medium"),
        ("Account locked after too many attempts", "My account was locked after a few failed login attempts. Please unlock it.", "Medium"),
        ("Cannot invite teammate", "I cannot invite a teammate - the invite button is greyed out.", "Medium"),
        ("Lost access to recovery codes", "I lost my recovery codes and cannot disable 2FA. Help me regain access.", "High"),
    ],
    "Shipping": [
        ("Package never arrived", "My order #10293 was supposed to arrive 10 days ago and never arrived. Where is it?", "Medium"),
        ("Wrong item delivered", "I received the wrong item in my order. I need a replacement shipped urgently.", "High"),
        ("Tracking number not updating", "The tracking number has not updated in 5 days. Is my package lost?", "Medium"),
        ("Damaged package", "My package arrived damaged and the product is broken. I want a replacement.", "High"),
        ("Change shipping address", "Can I change the shipping address for order #9821 before it ships?", "Low"),
        ("Late delivery - urgent", "This is urgent - my delivery is 2 weeks late and I needed it for an event yesterday.", "Urgent"),
        ("Missing items in shipment", "Two items are missing from my shipment. Please send the missing items.", "Medium"),
        ("Return label request", "How do I get a return shipping label for order #5512?", "Low"),
    ],
    "Other": [
        ("Feature request: dark mode", "It would be great to have a dark mode option in the app.", "Low"),
        ("Feedback on new UI", "The new UI looks clean but I miss the old keyboard shortcuts.", "Low"),
        ("Question about enterprise plan", "Can someone contact me about pricing for the enterprise plan?", "Medium"),
        ("Partnership inquiry", "We are interested in a partnership. Who should we talk to?", "Low"),
        ("General complaint", "I am disappointed with the recent changes to the refund policy.", "Medium"),
        ("Documentation outdated", "The API documentation is outdated and examples do not work.", "Low"),
        ("Compliment for support", "Just wanted to say the support team was very helpful last week. Thank you!", "Low"),
        ("Legal question", "We have a legal question about your data processing addendum.", "Medium"),
    ],
}


def _random_timestamp(start: datetime, days: int = 30) -> str:
    delta = timedelta(seconds=random.randint(0, days * 86400))
    return (start + delta).strftime("%Y-%m-%d %H:%M:%S")


def generate(n: int = 300, out: Path = settings.TICKETS_CSV) -> Path:
    start = datetime(2026, 10, 22, 8, 0, 0)
    rows = []
    # Weighted category mix so the distribution is realistic but not uniform.
    weights = {"Billing": 5, "Technical": 6, "Account": 4, "Shipping": 3, "Other": 2}
    pool = []
    for cat, w in weights.items():
        pool.extend([cat] * w)

    for i in range(1, n + 1):
        cat = random.choice(pool)
        subject, body, _ = random.choice(TEMPLATES[cat])
        customer = random.choice(CUSTOMERS) + str(random.randint(1, 50))
        rows.append({
            "id": i,
            "received_at": _random_timestamp(start),
            "customer": customer,
            "subject": subject,
            "body": body,
        })

    # Deliberately inject a few edge cases for robustness testing:
    # 2 empty bodies, 1 empty subject, 1 duplicate id.
    rows[150]["body"] = ""
    rows[151]["body"] = ""
    rows[200]["subject"] = ""
    dup = dict(rows[299])
    dup["body"] = dup["body"] + " (duplicate id test)"
    rows.append(dup)  # id 300 duplicated -> total 301 rows, 300 unique ids

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows ({df['id'].nunique()} unique ids) -> {out}")
    return out


if __name__ == "__main__":
    generate()
