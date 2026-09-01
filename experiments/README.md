# Circle-tracking experiment

Run two circle loops headlessly for both hands and write the desired and
measured pen-tip positions to CSV:

```bash
python run_circle_tracking.py
python run_circle_tracking.py --loops 6 --hand shadow_hand
```

The default outputs are
`results/shadow_hand_circle_tracking.csv` and
`results/wuji_hand_circle_tracking.csv`. Samples are written at the
controller rate (50 Hz); use `--sample-every 1` to record every MuJoCo step.

Compute RMSE, mean, median, 95th-percentile, maximum and per-axis RMSE, then
create a 3D trajectory plot and error-over-time plot for each hand:

```bash
python plot_circle_tracking.py
```

The numerical summary is saved to
`results/circle_tracking_stats.csv`. Pass `--help` to either script
for hand selection, alternate loop counts, input paths, and output locations.
