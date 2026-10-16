"""Compare two benchmark runs (baseline vs candidate) at each shared concurrency.

    python scripts/compare.py benchmarks/seqs16 benchmarks/seqs64 > benchmarks/seqs16-vs-64.md

Refuses to compare runs whose load shape differs, since that would not be like-for-like.
"""

import argparse
import json
from pathlib import Path

SHAPE = ("max_tokens", "prompt_words", "stream", "rounds", "min_requests", "models")
METRICS = (
    ("output_tokens_per_second", "Out tok/s", 1, 0, True),
    ("latency_p95", "p95 latency (s)", 1, 2, False),
    ("latency_p99", "p99 latency (s)", 1, 2, False),
    ("ttft_p95", "TTFT p95 (ms)", 1000, 0, False),
    ("gpu_mean_util_pct", "GPU util %", 1, 1, True),
    ("error_rate", "Err %", 100, 1, False),
)


def load(directory):
    d = Path(directory)
    return json.loads((d / "manifest.json").read_text()), json.loads(
        (d / "summary.json").read_text()
    )


def cell(a, b, scale, digits):
    if a is None or b is None:
        return "–"
    change = f" ({(b - a) / a:+.0%})" if a else ""
    return f"{a * scale:.{digits}f} → {b * scale:.{digits}f}{change}"


def compare(base_dir, cand_dir):
    (bm, brows), (cm, crows) = load(base_dir), load(cand_dir)
    mismatched = [k for k in SHAPE if bm.get(k) != cm.get(k)]
    if mismatched:
        raise ValueError(f"Load shape differs on {mismatched}; not a like-for-like comparison")
    if "mock" in (bm["backend"], cm["backend"]):
        raise ValueError("Refusing to compare mock runs")
    b = {r["concurrency"]: r for r in brows}
    c = {r["concurrency"]: r for r in crows}
    shared = sorted(set(b) & set(c))
    changed = {k: (bm["env"].get(k), cm["env"].get(k)) for k in set(bm["env"]) | set(cm["env"])}
    changed = {k: v for k, v in changed.items() if v[0] != v[1]}
    lines = [
        f"# {bm['label']} → {cm['label']}",
        "",
        f"Changed settings: {json.dumps(changed) if changed else 'none recorded in env'}",
        f"Commits: `{bm['git_commit']}` → `{cm['git_commit']}`",
        "",
        "| Conc. | " + " | ".join(m[1] for m in METRICS) + " |",
        "|---:|" + "---|" * len(METRICS),
    ]
    for k in shared:
        lines.append(
            f"| {k} | "
            + " | ".join(cell(b[k][m], c[k][m], s, d) for m, _, s, d, _ in METRICS)
            + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("candidate")
    a = p.parse_args()
    print(compare(a.baseline, a.candidate), end="")
