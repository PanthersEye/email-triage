from textwrap import dedent

def draft(subject: str, body: str, category: str, priority_label: str) -> str:
    if category == "support":
        core = "I understand the issue and I’m here to help."
    elif category == "sales":
        core = "Happy to share pricing and next steps."
    elif category == "hr":
        core = "Thanks for reaching out—see details and next steps below."
    else:
        core = "Thanks for the note—sharing details and next steps below."

    return dedent(f"""Hi,

Thanks for the details about: {subject or "(no subject)"}.
{core}

Could you share any additional context or screenshots if available?

Best,
Support Team
""")
