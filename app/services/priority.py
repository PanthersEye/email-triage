import re

URGENCY_KEYWORDS = [
    "urgent","asap","immediately","important","help",
    "emergency","issue","crash","broken","can't access",
    "need","problem",
]

def priority_score(text: str) -> float:
    t = text.lower()
    hits = 0
    for kw in URGENCY_KEYWORDS:
        if kw == "can't access":
            if "can't access" in t or "cant access" in t:
                hits += 1
        elif re.search(rf"\b{re.escape(kw)}\b", t):
            hits += 1
    return min(hits / len(URGENCY_KEYWORDS), 1.0)

def priority_label(score: float) -> str:
    if score > 0.30:
        return "High"
    if score > 0.10:
        return "Medium"
    return "Low"
