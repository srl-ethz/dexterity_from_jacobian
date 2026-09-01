#!/usr/bin/env python3
"""Compute circle-tracking statistics and plot trajectory and error."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
POSITION_COLUMNS = {
    "desired": ("desired_x_m", "desired_y_m", "desired_z_m"),
    "measured": ("measured_x_m", "measured_y_m", "measured_z_m"),
}


def load_tracking_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"CSV has no samples: {path}")
    data = np.atleast_1d(data)
    required = {
        "tracking_time_s",
        *POSITION_COLUMNS["desired"],
        *POSITION_COLUMNS["measured"],
    }
    missing = required.difference(data.dtype.names or ())
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return data


def _positions(data, kind):
    return np.column_stack([data[column] for column in POSITION_COLUMNS[kind]])


def compute_stats(data):
    desired = _positions(data, "desired")
    measured = _positions(data, "measured")
    error = measured - desired
    error_norm = np.linalg.norm(error, axis=1)
    duration = float(data["tracking_time_s"][-1] - data["tracking_time_s"][0])
    return {
        "samples": len(data),
        "duration_s": duration,
        "rmse_m": float(np.sqrt(np.mean(error_norm**2))),
        "mean_error_m": float(np.mean(error_norm)),
        "median_error_m": float(np.median(error_norm)),
        "p95_error_m": float(np.percentile(error_norm, 95)),
        "max_error_m": float(np.max(error_norm)),
        "rmse_x_m": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_y_m": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "rmse_z_m": float(np.sqrt(np.mean(error[:, 2] ** 2))),
    }


def _set_equal_3d_limits(axis, points):
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 0.5)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plot_trajectory(data, output_path, title):
    desired = _positions(data, "desired")
    measured = _positions(data, "measured")
    center = desired.mean(axis=0)
    desired_mm = (desired - center) * 1e3
    measured_mm = (measured - center) * 1e3

    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*desired_mm.T, label="Desired", color="tab:blue", linewidth=2.0)
    axis.plot(*measured_mm.T, label="Measured", color="tab:orange", linewidth=1.2)
    axis.scatter(*desired_mm[0], color="black", marker="o", s=25, label="Start")
    axis.set_xlabel("x offset [mm]")
    axis.set_ylabel("y offset [mm]")
    axis.set_zlabel("z offset [mm]")
    axis.set_title(f"{title}: pen-tip trajectory")
    axis.legend()
    _set_equal_3d_limits(axis, np.vstack((desired_mm, measured_mm)))
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_error(data, output_path, title, stats, y_limits):
    desired = _positions(data, "desired")
    measured = _positions(data, "measured")
    error_mm = (measured - desired) * 1e3
    error_norm_mm = np.linalg.norm(error_mm, axis=1)
    time = data["tracking_time_s"] - data["tracking_time_s"][0]

    figure, axis = plt.subplots(figsize=(4, 2.5))
    axis.plot(time, error_norm_mm, color="black", linewidth=1.3, label=r"$||e||_2$")
    for index, coordinate in enumerate("xyz"):
        axis.plot(
            time,
            error_mm[:, index],
            linewidth=0.8,
            alpha=0.65,
            label=rf"$e_{coordinate}$",
        )
    rmse_mm = stats["rmse_m"] * 1e3
    axis.axhline(
        rmse_mm,
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
    )
    axis.text(
        0.98,
        rmse_mm,
        f"RSME={rmse_mm:.2f}mm",
        color="tab:red",
        ha="right",
        va="bottom",
        transform=axis.get_yaxis_transform(),
    )
    axis.set_ylim(y_limits)
    axis.set_xlabel("tracking time [s]")
    axis.set_ylabel("tracking error [mm]")
    axis.set_title(f"{title}: tracking error over time")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyse_file(csv_path, data, output_dir, error_y_limits):
    stats = compute_stats(data)
    title = csv_path.stem.replace("_", " ").title()
    stem = csv_path.stem
    plot_trajectory(data, output_dir / f"{stem}_trajectory_3d.png", title)
    plot_error(
        data,
        output_dir / f"{stem}_error_over_time.pdf",
        title,
        stats,
        error_y_limits,
    )
    return {"dataset": stem, **stats}


def _shared_error_y_limits(datasets):
    error_values_mm = []
    for data in datasets:
        error_mm = (_positions(data, "measured") - _positions(data, "desired")) * 1e3
        error_values_mm.extend((error_mm.ravel(), np.linalg.norm(error_mm, axis=1)))

    low = min(float(values.min()) for values in error_values_mm)
    high = max(float(values.max()) for values in error_values_mm)
    margin = max((high - low) * 0.05, 0.1)
    return low - margin, high + margin


def write_summary(rows, path):
    if not rows:
        return
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nPerformance summary")
    print(
        f"{'dataset':34s} {'RMSE [mm]':>10s} {'mean [mm]':>10s} "
        f"{'p95 [mm]':>10s} {'max [mm]':>10s}"
    )
    for row in rows:
        print(
            f"{row['dataset']:34s} "
            f"{row['rmse_m'] * 1e3:10.3f} "
            f"{row['mean_error_m'] * 1e3:10.3f} "
            f"{row['p95_error_m'] * 1e3:10.3f} "
            f"{row['max_error_m'] * 1e3:10.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="tracking CSVs (default: all *_circle_tracking.csv in experiments/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"plot and summary directory (default: {DEFAULT_RESULTS_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_files = args.csv_files or sorted(
        DEFAULT_RESULTS_DIR.glob("*_circle_tracking.csv")
    )
    if not csv_files:
        raise SystemExit(
            "No tracking CSVs found. Run experiments/run_circle_tracking.py first "
            "or pass CSV paths explicitly."
        )
    missing = [path for path in csv_files if not path.is_file()]
    if missing:
        raise SystemExit(f"CSV file not found: {missing[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_tracking_csv(path) for path in csv_files]
    error_y_limits = _shared_error_y_limits(datasets)
    rows = [
        analyse_file(path, data, args.output_dir, error_y_limits)
        for path, data in zip(csv_files, datasets)
    ]
    summary_path = args.output_dir / "circle_tracking_stats.csv"
    write_summary(rows, summary_path)
    print_summary(rows)
    print(f"\nWrote plots and {summary_path}")


if __name__ == "__main__":
    main()
