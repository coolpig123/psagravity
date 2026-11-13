"""Render PROGRESS.md from scripts/progress.json and print the progress bar.

    python scripts/progress.py

Progress % = sum of estimates of finished steps (+ half of the running step) / sum of estimates.
Times are in minutes in the JSON.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "scripts" / "progress.json"
ICON = {"done": "✅", "running": "🔄", "todo": "⏳", "blocked": "⛔"}


def hm(m: float | None) -> str:
    if m is None:
        return ""
    m = int(round(m))
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def main():
    s = json.loads(STATE.read_text())
    now = datetime.now(timezone.utc)
    started = datetime.fromisoformat(s["started_utc"])
    elapsed = (now - started).total_seconds() / 60
    steps = s["steps"]
    est_total = sum(st["est"] for st in steps)
    done_w = sum(st["est"] for st in steps if st["status"] == "done") + \
        sum(st["est"] / 2 for st in steps if st["status"] == "running")
    pct = round(100 * done_w / est_total)
    # remaining: estimates of unfinished steps (running step: its estimate minus time already spent, floor 10 min)
    left = sum(st["est"] for st in steps if st["status"] == "todo") + \
        sum(max(10, st["est"] - (st.get("actual") or 0)) for st in steps if st["status"] == "running")
    total_proj = elapsed + left
    bar = "█" * round(pct / 100 * 16) + "░" * (16 - round(pct / 100 * 16))
    head = (f"Overall: [{bar}] {pct}%   elapsed {hm(elapsed)} / est. total ~{hm(total_proj)}  → ~{hm(left)} left\n"
            f"Last update: {now:%Y-%m-%d %H:%M} UTC")
    lines = ["# Progress: more fires (Crosswhite, Colwash, Biscar) + summary, website, README", ""]
    if s.get("pr"):
        lines += [f"**Pull request:** {s['pr']}", ""]
    lines += ["```", head, "```", "",
              "| # | Step | Status | Est. | Actual |", "|---|---|---|---|---|"]
    for i, st in enumerate(steps, 1):
        lines.append(f"| {i} | {st['name']} | {ICON[st['status']]} | {hm(st['est'])} | {hm(st.get('actual'))} |")
    for title, key in [("Done so far", "done"), ("Problems / data mismatches", "problems"),
                       ("Open questions for Ori", "questions")]:
        lines += ["", f"## {title}", ""]
        lines += [f"- {x}" for x in s.get(key, [])] or ["- (none yet)"]
    (ROOT / "PROGRESS.md").write_text("\n".join(lines) + "\n")
    print(head)


if __name__ == "__main__":
    main()
