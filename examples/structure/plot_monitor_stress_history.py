#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


POINTS = ("A", "B", "C")


def read_history(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        data = {field: [] for field in reader.fieldnames or []}
        for row in reader:
            for field in data:
                data[field].append(float(row[field]))
    if not data:
        raise ValueError(f"No data found in {csv_path}")
    return data


def plot_signed_tau33(data: dict[str, list[float]], output_path: Path) -> None:
    time_ms = [value * 1.0e3 for value in data["t_s"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for point in POINTS:
        tau33_mpa = [value / 1.0e6 for value in data[f"tau33_{point}_Pa"]]
        ax.plot(time_ms, tau33_mpa, linewidth=2, label=f"Point {point}")

    ax.set_title("Axial Kirchhoff Stress History")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(r"$\tau_{33}$ [MPa]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_von_mises(data: dict[str, list[float]], output_path: Path) -> None:
    time_ms = [value * 1.0e3 for value in data["t_s"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for point in POINTS:
        tauvm_mpa = [value / 1.0e6 for value in data[f"tauvm_{point}_Pa"]]
        ax.plot(time_ms, tauvm_mpa, linewidth=2, label=f"Point {point}")

    ax.set_title("Von Mises Stress History")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(r"$\tau_{vm}$ [MPa]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_combined_panels(data: dict[str, list[float]], output_path: Path) -> None:
    time_ms = [value * 1.0e3 for value in data["t_s"]]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for point in POINTS:
        axes[0].plot(time_ms,
                     [value / 1.0e6 for value in data[f"tau33_{point}_Pa"]],
                     linewidth=2,
                     label=f"Point {point}")
        axes[1].plot(time_ms,
                     [value / 1.0e6 for value in data[f"tauvm_{point}_Pa"]],
                     linewidth=2,
                     label=f"Point {point}")

    axes[0].set_title("Stress History at Monitoring Points")
    axes[0].set_ylabel(r"$\tau_{33}$ [MPa]")
    axes[1].set_ylabel(r"$\tau_{vm}$ [MPa]")
    axes[1].set_xlabel("Time [ms]")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sandstone monitor stress history CSV.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/home/sadaat/SPH_Code/TrixiParticles.jl_Saadat/out_sandstone_uniaxial/monitor_stress_history.csv",
        help="Path to monitor_stress_history.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where PNG files will be written. Defaults to the CSV directory.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_history(csv_path)

    plot_signed_tau33(data, output_dir / "tau33_history.png")
    plot_von_mises(data, output_dir / "tauvm_history.png")
    plot_combined_panels(data, output_dir / "monitor_stress_history_combined.png")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()