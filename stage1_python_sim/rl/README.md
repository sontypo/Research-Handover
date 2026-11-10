# Reinforcement learning with stable-baselines3

A minimal, working pipeline: train → evaluate → export → run in 2-D, MuJoCo, Gazebo
or on the robot. It is a generic starting point (MLP + PPO + a standard reward), meant
to be replaced by your own design.

| Script | Does |
|---|---|
| `train_sb3.py` | trains PPO (or SAC) on `socialnav_core.gym_env.SocialNavGymEnv`, evaluates on the 4 benchmark scenarios every 100k steps, keeps `best_model.zip`, exports it |
| `export_policy.py` | SB3 model → `policy.pt` (TorchScript, torch only), `policy.npz` (numpy only), `policy_meta.json`; verifies that all formats give the same actions |
| `play.py` | metrics, trajectory plots and GIFs of any exported policy in the 2-D simulator |

```bash
python3 train_sb3.py --timesteps 1500000 --run-name ppo_mixed      # ≈ 10-15 min on 16 cores
python3 play.py --policy runs/ppo_mixed/policy.pt --gif
python3 ../run_benchmark.py --policy runs/ppo_mixed/policy.pt       # vs. the baselines
```

## What the policy sees and does

* **Observation** (`socialnav_core.observation.to_vector`, 69 values by default): goal
  distance and bearing, own speed and yaw rate, the 8 nearest people (relative position
  and velocity in the robot frame), and 16 range rays (a coarse 2-D LiDAR that sees
  walls).
* **Action**: two numbers in [-1, 1] mapped to forward speed [0, 1] m/s and yaw rate
  [-1.2, 1.2] rad/s, every 0.1 s, through the acceleration limits of the robot.
* **Reward** (`socialnav_core.env.RewardWeights`): progress towards the goal, +10 at
  the goal, −10 on collision, a penalty when closer than 0.3 m to a person, and a small
  time penalty.
* **Training crowds**: the four benchmark layouts plus `random_crowd` with 2–12 people;
  people **never yield** to the robot unless `--robot-visible`.

## Deployment contract

`policy_meta.json` stores the control period, the observation layout and the action
scaling. Every runner reads it: `LearnedPolicy` in 2-D, `stage2_mujoco/mujoco_crowd.py`,
and the ROS node `nav_controller` (`controller:=learned policy:=.../policy.pt`), which
also sets its control rate from it and builds the 16 rays from `/scan`. `policy.pt`
needs only torch; `policy.npz` is the same actor for machines without torch.

## Ideas for your own work

Change the encoder (attention or graph networks over people), the reward, the
curriculum, the algorithm, or the observation; compare with the same seeds and
`run_benchmark.py`. Keep the export contract, and every runner keeps working.
