import csv, time, pathlib, os

from dataclasses import dataclass, asdict
file_path = "data/cost_log.csv"
LOG = pathlib.Path(file_path)
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
    tag: str


def log_call(model: str, in_tokens: int, out_tokens: int, latency_ms: float, tag: str):
    p = PRICES[model]
    in_cost = p["in"] * in_tokens
    print(in_cost)
    out_cost = p['out'] * out_tokens
    cost_usd = in_cost + out_cost
    # New row to be added in the log
    new_row = CallLog(ts=time.time(), model=model, in_tokens=in_tokens, out_tokens=out_tokens, in_cost=in_cost, out_cost=out_cost, cost_usd=cost_usd, latency_ms=latency_ms, tag=tag)

    # Open the file in the append mode
    with LOG.open(mode="a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=asdict(new_row).keys())
        print(w)
        if not LOG.exists or os.path.getsize(file_path) == 0:
            print("inside")
            w.writeheader()
        w.writerow(asdict(new_row))
    return cost_usd

print(log_call("gpt-5-nano", 19, 20, 11, "test"))