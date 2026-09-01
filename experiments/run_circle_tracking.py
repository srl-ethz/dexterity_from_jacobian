#!/usr/bin/env python3
"""Run headless circle-tracking experiments for the Shadow and Wuji hands."""

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sim import CONTROL_DECIMATION, CIRCLE_ANGULAR_SPEED, create_shadow_hand_sim  # noqa: E402
from sim_wuji_hand_2 import create_wuji_hand_sim  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
HAND_FACTORIES = {
    "shadow_hand": create_shadow_hand_sim,
    "wuji_hand": create_wuji_hand_sim,
}
CSV_FIELDS = (
    "hand",
    "sample",
    "simulation_time_s",
    "tracking_time_s",
    "loop_progress",
    "desired_x_m",
    "desired_y_m",
    "desired_z_m",
    "measured_x_m",
    "measured_y_m",
    "measured_z_m",
    "error_x_m",
    "error_y_m",
    "error_z_m",
    "error_norm_m",
)


def _sample_row(hand, sample, data, controller):
    desired, _ = controller.circle_reference(data.time)
    measured = data.xpos[controller.pen_tip_id].copy()
    error = measured - desired
    tracking_time = max(data.time - controller.tracking_start_time, 0.0)
    return {
        "hand": hand,
        "sample": sample,
        "simulation_time_s": data.time,
        "tracking_time_s": tracking_time,
        "loop_progress": (
            CIRCLE_ANGULAR_SPEED * tracking_time / (2.0 * np.pi)
        ),
        "desired_x_m": desired[0],
        "desired_y_m": desired[1],
        "desired_z_m": desired[2],
        "measured_x_m": measured[0],
        "measured_y_m": measured[1],
        "measured_z_m": measured[2],
        "error_x_m": error[0],
        "error_y_m": error[1],
        "error_z_m": error[2],
        "error_norm_m": np.linalg.norm(error),
    }


def run_hand(hand, output_path, loops=3.0, sample_every=CONTROL_DECIMATION):
    """Run one hand through bootstrap and ``loops`` reference circles."""
    if hand not in HAND_FACTORIES:
        raise ValueError(f"Unknown hand: {hand}")
    if loops <= 0:
        raise ValueError("loops must be positive")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")

    model, data, controller = HAND_FACTORIES[hand]()
    simulation_dt = float(model.opt.timestep)
    tracking_duration = loops * 2.0 * np.pi / CIRCLE_ANGULAR_SPEED
    # A circle period is not generally an integer number of MuJoCo steps. Use
    # ceil so the recorded run always completes at least the requested loops.
    tracking_steps = int(np.ceil(tracking_duration / simulation_dt))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {hand} excitation, "
        f"{tracking_duration:.3f} s tracking ({loops:g} loops)"
    )
    mujoco.set_mjcb_control(controller.control_cb)
    try:
        while data.time + simulation_dt / 2.0 < controller.tracking_start_time:
            mujoco.mj_step(model, data)

        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            sample = 0
            writer.writerow(_sample_row(hand, sample, data, controller))

            for step in range(1, tracking_steps + 1):
                mujoco.mj_step(model, data)
                if step % sample_every == 0 or step == tracking_steps:
                    sample += 1
                    writer.writerow(_sample_row(hand, sample, data, controller))
    finally:
        # MuJoCo's control callback is process-global, so never leave one hand's
        # controller installed when constructing or running the next hand.
        mujoco.set_mjcb_control(None)

    print(f"Wrote {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hands",
        nargs="+",
        choices=tuple(HAND_FACTORIES),
        default=tuple(HAND_FACTORIES),
        help="hands to simulate (default: both)",
    )
    parser.add_argument(
        "--loops",
        type=float,
        default=2.0,
        help="number of reference-circle loops (default: 3)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=CONTROL_DECIMATION,
        help=(
            "write one row per N simulation steps "
            f"(default: {CONTROL_DECIMATION}, matching the controller rate)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CSV output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for hand in args.hands:
        run_hand(
            hand,
            args.output_dir / f"{hand}_circle_tracking.csv",
            loops=args.loops,
            sample_every=args.sample_every,
        )


if __name__ == "__main__":
    main()
