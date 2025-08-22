import math

def score(subject: str, body: str, category: str):
    t = f"{subject or ''} {body or ''}".lower()
    w = 0.0
    for kw in ["urgent","asap","immediately","outage","down","cannot","can't","failure"]:
        if kw in t: w += 1.5
    for kw in ["enterprise","invoice","refund","contract","renewal","vip"]:
        if kw in t: w += 0.8
    if category == "support": w += 0.5
    if "login" in t: w += 0.7
    w = max(0.0, min(4.0, w))
    s = 1/(1+math.exp(-w))
    label = "High" if s >= 0.75 else ("Medium" if s >= 0.4 else "Low")
    return float(s), label
