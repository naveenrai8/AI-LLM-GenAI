import csv, time, pathlib, os
from pathlib import Path
from datetime import datetime
from find_root_directory import get_project_root

from dataclasses import dataclass, asdict
BASE_DIR = get_project_root()
LOG = BASE_DIR / "data" / "cost_log.csv"
LOG.parent.mkdir(exist_ok=True)

PRICES = {
    "gpt-5-nano": {
        "in": 0.05 / 1_000_000, "out": 0.40 / 1_000_000
    },
    "text-embedding-3-large": {
        "in": 0.000143 / 1_000_000, "out": 0
    },
    "text-embedding-3-small": {
        "in": 0.000022 / 1_000_000, "out": 0
    },
    "gpt-5-mini": {
        "in": 0.28 / 1_000_000, "out": 2 / 1_000_000
    }
}

@dataclass
class CallLog:
    ts: float
    model: str
    in_tokens: int
    out_tokens: int
    in_cost: float
    out_cost: float
    cost_usd: float
    latency_ms: float
    time_to_first_token: float
    tag: str


def log_call(model: str, in_tokens: int, out_tokens: int, latency_ms: float, time_to_first_token: float, tag: str):
    p = PRICES[model]
    in_cost = p["in"] * in_tokens
    print(in_cost)
    out_cost = p['out'] * out_tokens
    cost_usd = in_cost + out_cost
    if not time_to_first_token:
        time_to_first_token = latency_ms
    # New row to be added in the log
    timestamp= time.time()
    new_row = CallLog(ts=datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'), model=model, in_tokens=in_tokens, out_tokens=out_tokens, in_cost=in_cost, out_cost=out_cost, cost_usd=cost_usd, latency_ms=latency_ms, time_to_first_token=time_to_first_token, tag=tag)

    # Open the file in the append mode
    with LOG.open(mode="a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=asdict(new_row).keys())
        print(w)
        if not LOG.exists() or LOG.stat().st_size == 0:
            print("inside")
            w.writeheader()
        w.writerow(asdict(new_row))
    return cost_usd