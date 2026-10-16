"""Background nvidia-smi sampler: per-GPU utilization and memory while a run is in flight.

Must run on the GPU host. If nvidia-smi is missing it reports {"available": false} rather
than inventing numbers.
"""

import shutil
import subprocess
import threading
import time

QUERY = "index,name,utilization.gpu,memory.used,memory.total"


def query():
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        index, name, util, used, total = (part.strip() for part in line.split(","))
        rows.append(
            {
                "index": int(index),
                "name": name,
                "util_pct": float(util),
                "memory_used_mib": float(used),
                "memory_total_mib": float(total),
            }
        )
    return rows


def describe():
    """Static host facts for a benchmark manifest."""
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    ).stdout
    return {"available": True, "gpus": [line.strip() for line in out.strip().splitlines()]}


class GpuSampler:
    def __init__(self, interval=0.5, source=None):
        self.interval = interval
        self.source = source or (query if shutil.which("nvidia-smi") else None)
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.source:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self.source())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval)

    def summary(self):
        if not self.source or not self.samples:
            return {"available": False}
        gpus = {}
        for sample in self.samples:
            for row in sample:
                gpus.setdefault(row["index"], {"name": row["name"], "util": [], "mem": []})
                gpus[row["index"]]["util"].append(row["util_pct"])
                gpus[row["index"]]["mem"].append(row["memory_used_mib"])
                gpus[row["index"]]["total"] = row["memory_total_mib"]
        return {
            "available": True,
            "samples": len(self.samples),
            "interval_seconds": self.interval,
            "gpus": [
                {
                    "index": i,
                    "name": g["name"],
                    "mean_util_pct": round(sum(g["util"]) / len(g["util"]), 1),
                    "peak_memory_mib": max(g["mem"]),
                    "memory_total_mib": g["total"],
                }
                for i, g in sorted(gpus.items())
            ],
            "note": "nvidia-smi utilization is the fraction of time a kernel was running, "
            "not SM occupancy. vLLM preallocates KV cache, so memory reflects "
            "--gpu-memory-utilization more than live load.",
        }


if __name__ == "__main__":
    import json

    with GpuSampler() as sampler:
        time.sleep(2)
    print(json.dumps({"host": describe(), "sample": sampler.summary()}, indent=2))
