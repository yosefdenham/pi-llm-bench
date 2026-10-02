import json
import statistics
from pathlib import Path

RAW = Path.home() / "bench" / "data" / "raw"


def test_label(entry):
    """llama-bench does not name the test; derive it from n_prompt/n_gen."""
    if entry["n_gen"] == 0:
        return f"pp{entry['n_prompt']}"
    return f"tg{entry['n_gen']}"


def load(path):
    with open(path) as f:
        entries = json.load(f)
    rows = []
    for e in entries:
        samples = e["samples_ts"]
        rows.append({
            "model": Path(e["model_filename"]).name,
            "size_gib": e["model_size"] / (1024 ** 3),
            "threads": e["n_threads"],
            "test": test_label(e),
            "mean": e["avg_ts"],
            "median": statistics.median(samples),
            "stddev": e["stddev_ts"],
            "samples": samples,
        })
    return rows


def show(rows, title, key):
    print(f"\n=== {title} ===")
    print(f"{'config':<40} {'test':>7} {'median t/s':>11} {'stddev':>8} {'GiB/s':>7}")
    for r in sorted(rows, key=key):
        bw = r["size_gib"] * r["median"] if r["test"].startswith("tg") else 0
        bw_str = f"{bw:.2f}" if bw else ""
        label = f"{r['model'][:30]:<30} t={r['threads']}"
        print(f"{label:<40} {r['test']:>7} {r['median']:>11.2f} "
              f"{r['stddev']:>8.3f} {bw_str:>7}")


threads = load(RAW / "threads.json")
quants = load(RAW / "quants.json")

show(threads, "Thread scaling (TinyLlama Q4_K_M)",
     lambda r: (r["test"], r["threads"]))
show(quants, "Quantization sweep (4 threads)",
     lambda r: (r["test"], r["size_gib"]))
