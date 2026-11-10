# Stage 1 — 2-D simulation

```bash
python3 run_benchmark.py                 # 4 scenarios x 3 baselines x 20 seeds -> outputs/
python3 run_benchmark.py --robot-visible # people react to the robot (easier)
python3 animate.py --scenario chokepoint --controller social_force --seed 3
python3 animate.py --all                 # one GIF per scenario and baseline
python3 train_policy_template.py         # interface check for your learned policy
```

Outputs: `results.csv` (one row per episode), `summary.md`, `summary.png`,
`trajectories_<scenario>.png`, `gifs/`. Add `--save-logs` to keep raw episodes (`.npz`)
for prediction experiments. The library lives in `../core/socialnav_core`.
