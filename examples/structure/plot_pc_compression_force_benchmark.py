import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(csv_path: Path):
    grouped = defaultdict(lambda: {"time_s": [], "force_N": [], "temp_C": None, "speed_mm_s": None})
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = row["case_id"]
            grouped[case_id]["time_s"].append(float(row["time_s"]))
            grouped[case_id]["force_N"].append(float(row["force_N"]))
            grouped[case_id]["temp_C"] = float(row["temp_C"])
            grouped[case_id]["speed_mm_s"] = float(row["speed_mm_s"])
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Plot PC compression-force benchmark CSV.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/home/sadaat/SPH_Code/TrixiParticles.jl_Saadat/out_pc_compression_benchmark/compression_force_history.csv",
        help="Path to compression_force_history.csv",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for PNG output.")
    parser.add_argument("--speed", type=float, default=0.01, help="Speed in mm/s to plot.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = read_history(csv_path)
    selected = [entry for entry in grouped.values() if abs(entry["speed_mm_s"] - args.speed) < 1e-12]
    if not selected:
        raise SystemExit(f"No curves found for speed {args.speed} mm/s in {csv_path}")

    selected.sort(key=lambda item: item["temp_C"])

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    for entry in selected:
        ax.plot(entry["time_s"], entry["force_N"], linewidth=2.0, label=f"{entry['temp_C']:.0f} C")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Compression force (N)")
    ax.set_title(f"PC Compression Force Benchmark, {args.speed:g} mm/s")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / f"pc_compression_force_{args.speed:g}mm_s.png"
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()