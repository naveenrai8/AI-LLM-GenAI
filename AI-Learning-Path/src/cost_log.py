"""Cost + latency logger for every LLM call, plus plotting helpers.

Non-negotiable rule (see month-0.md): every LLM call in every notebook calls `log_call`.
"""

from __future__ import annotations

import csv
import pathlib
import time
from dataclasses import asdict, dataclass

LOG = pathlib.Path("data/cost_log.csv")
LOG.parent.mkdir(exist_ok=True)

# Azure OpenAI Oct 2024 prices — verify at
# https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
PRICES: dict[str, dict[str, float]] = {
    "gpt-4o-mini":            {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    "gpt-4o":                 {"in": 2.50 / 1_000_000, "out": 10.00 / 1_000_000},
    "text-embedding-3-small": {"in": 0.02 / 1_000_000, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13 / 1_000_000, "out": 0.0},
    # Local / self-hosted: zero $ but still log tokens
    "ollama-local":           {"in": 0.0, "out": 0.0},
    "vllm-self-hosted":       {"in": 0.0, "out": 0.0},
}


@dataclass
class CallLog:
    ts: float
    model: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    latency_ms: float
    tag: str


def log_call(model: str, in_tokens: int, out_tokens: int, latency_ms: float, tag: str) -> float:
    """Append one row to data/cost_log.csv. Returns the computed cost."""
    p = PRICES.get(model)
    if p is None:
        raise ValueError(f"Unknown model '{model}'. Add it to PRICES in src/cost_log.py.")
    cost = in_tokens * p["in"] + out_tokens * p["out"]
    row = CallLog(time.time(), model, in_tokens, out_tokens, cost, latency_ms, tag)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=asdict(row).keys())
        if new:
            w.writeheader()
        w.writerow(asdict(row))
    return cost


# ----------------------------------------------------------------------
# Plotting + reporting helpers (imported lazily so the logger stays light)
# ----------------------------------------------------------------------

def _load(tags=None, since=None, until=None):
    """Load cost_log.csv into a DataFrame, optionally filtered."""
    import pandas as pd

    df = pd.read_csv(LOG)
    df["ts"] = pd.to_datetime(df["ts"])
    if tags is not None:
        df = df[df["tag"].isin(tags)]
    if since is not None:
        df = df[df["ts"] >= pd.Timestamp(since)]
    if until is not None:
        df = df[df["ts"] <= pd.Timestamp(until)]
    return df


def summary(tags=None, since=None, until=None):
    """Return a tag-level summary DataFrame (calls, cost, latency, tokens)."""
    df = _load(tags, since, until)
    return (
        df.groupby("tag")
        .agg(
            calls=("cost_usd", "size"),
            total_cost=("cost_usd", "sum"),
            avg_cost=("cost_usd", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            in_tokens=("in_tokens", "sum"),
            out_tokens=("out_tokens", "sum"),
        )
        .sort_values("total_cost", ascending=False)
    )


def plot_costs(tags=None, since=None, until=None, figsize=(16, 4)):
    """Three-panel plot: total cost by tag, calls by tag, cumulative cost over time.

    Usage:
        from src.cost_log import plot_costs, summary
        plot_costs()
        plot_costs(since="2026-04-25")
        plot_costs(tags=["week-0.1", "week-0.2"])
        summary()
    """
    import matplotlib.pyplot as plt

    df = _load(tags, since, until)
    if df.empty:
        print("No rows match the filters.")
        return None

    cost_by_tag = df.groupby("tag")["cost_usd"].sum().sort_values()
    calls_by_tag = df.groupby("tag").size().sort_values()

    cum = (
        df.sort_values("ts")
        .assign(cum_cost=lambda d: d.groupby("tag")["cost_usd"].cumsum())
        .pivot_table(index="ts", columns="tag", values="cum_cost")
        .ffill()
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    cost_by_tag.plot(kind="barh", ax=axes[0], color="tab:blue")
    axes[0].set_title("Total cost per tag ($)")
    axes[0].set_xlabel("USD")

    calls_by_tag.plot(kind="barh", ax=axes[1], color="tab:orange")
    axes[1].set_title("Calls per tag")
    axes[1].set_xlabel("count")

    cum.plot(ax=axes[2])
    axes[2].set_title("Cumulative cost over time")
    axes[2].set_ylabel("USD")
    axes[2].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig


def plot_latency(tags=None, since=None, until=None, figsize=(10, 5)):
    """Boxplot of latency_ms per tag. Useful to spot slow experiments."""
    import matplotlib.pyplot as plt

    df = _load(tags, since, until)
    if df.empty:
        print("No rows match the filters.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    df.boxplot(column="latency_ms", by="tag", ax=ax, rot=30)
    ax.set_title("Latency distribution per tag (ms)")
    ax.set_ylabel("latency_ms")
    plt.suptitle("")  # drop pandas' auto title
    plt.tight_layout()
    plt.show()
    return fig


summary()
plot_costs()
plot_latency()
summary()