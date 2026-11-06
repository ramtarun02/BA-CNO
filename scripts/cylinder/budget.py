"""Day-2 budget: turn measured it/s into wall-clock hours per config and pick a step budget.

Reads ``timing.json`` written by ``train.py`` under each run directory. Every
config gets the same step budget (study plan section 6), so the budget is set by
the slowest config: the largest round number of steps that fits in
``--hours-per-config`` for it.

    python scripts/cylinder/budget.py --runs outputs/colab_timing/* --hours-per-config 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROUND = 25_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--hours-per-config", type=float, default=20.0, help="GPU hours you can spend on each config")
    ap.add_argument("--default", type=int, default=300_000, help="planned budget, shown for comparison")
    ap.add_argument("--out", default=None, help="write the table as JSON here")
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        f = Path(run) / "timing.json"
        if f.exists():
            rows.append({"run": Path(run).name, **json.loads(f.read_text())})
    if not rows:
        raise SystemExit("No timing.json found. Run train.py on each config first.")

    slowest = min(r["it_per_s"] for r in rows)
    budget = int(slowest * args.hours_per_config * 3600 // ROUND * ROUND)
    print(f"{'run':28s} {'device':8s} {'params':>10s} {'it/s':>7s} {'h @ budget':>11s} {'h @ default':>12s}")
    for r in rows:
        r["hours_at_budget"] = budget / r["it_per_s"] / 3600
        r["hours_at_default"] = args.default / r["it_per_s"] / 3600
        print(f"{r['run']:28s} {r['device']:8s} {r['params']:>10,d} {r['it_per_s']:7.2f} "
              f"{r['hours_at_budget']:11.1f} {r['hours_at_default']:12.1f}")
    total = sum(r["hours_at_budget"] for r in rows)
    print(f"\nbudget = {budget:,} steps per config ({args.hours_per_config:g} h for the slowest); "
          f"{total:.0f} GPU-h for all {len(rows)} configs")
    if args.out:
        Path(args.out).write_text(json.dumps({"budget_steps": budget, "hours_per_config": args.hours_per_config,
                                              "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
