# Concepts (minimal)

This page gives the ideas you need to start. It is deliberately short: follow the
references for depth.

## 1. The problem

A mobile robot must reach a goal through a space shared with people. Collision-free is
necessary but not sufficient: people expect personal space, predictable motion, and
that the robot does not cut through the groups they walk in. Two difficulties dominate:

* **Uncertainty about people.** Their goals and intentions are not observed; the robot
  only sees positions and velocities, with noise and occlusions.
* **The freezing robot problem.** If every person is treated as an obstacle that may go
  anywhere, a dense crowd makes every action look unsafe and the robot stops
  (Trautman & Krause, 2010).

A common formalisation is a POMDP: the robot observes the people it perceives, chooses a
velocity command every control step, and is rewarded for progress and penalised for
collisions and intrusions.

## 2. Crowd model used here: the Social Force Model

Each pedestrian is pushed towards its goal and repelled by others and walls
(Helbing & Molnár, 1995). `core/socialnav_core/sfm.py` implements it with textbook
parameters, anisotropy (people react to what is in front), and group cohesion.
It is a simple, controllable crowd, not a realistic one: use real datasets
(ETH/UCY, SDD) for prediction work.

**Non-yielding protocol.** By default the simulated people *ignore the robot*. This is
the honest test: a policy cannot succeed by walking into people and letting them step
aside. Use `--robot-visible` only to see the difference.

## 3. Three families of navigation methods

| Family | Idea | Here |
|---|---|---|
| Reactive | react to the current positions (Social Force, ORCA, DWA) | `SocialForceController` |
| Predict, then plan | forecast where people will be, then avoid the forecasts | `PredictiveSocialForce` + `ConstantVelocityPredictor` |
| Learned | learn the policy from experience (RL) and/or demonstrations (IL) | `LearnedPolicy` slot + `env.CrowdNavEnv` |

Ideas worth exploring, at the level of directions only:

* **Better forecasts.** Replace constant velocity by a learned, multimodal forecaster that
  models interactions between people, and keep it fast enough for the robot's control
  loop. Measure it with minADE/minFDE (`prediction.ade_fde`) before using it in control.
* **Social structure.** People walk in groups; a robot that understands groups can avoid
  splitting them. How groups are detected or learned is an open design choice.
* **Learning the policy.** RL in massively parallel simulation, with a reward that
  balances progress, safety and comfort; demonstrations can speed up the start of
  training. Watch for sim-to-real gaps (actuator lag, perception noise, frames).

## 4. How to measure

Report at least, per scenario and crowd size, over many seeds:

* success / collision / timeout rates (`metrics.episode_metrics`),
* time to goal of the successful episodes, path length,
* minimum centre distance to any person, and the exposure (share of time with a person
  closer than a stated threshold).

Always state the protocol (do people react to the robot?), the thresholds, and the
number of episodes. On a real robot, distances come from your own perception: say so.

## 5. Suggested reading

* D. Helbing, P. Molnár, "Social force model for pedestrian dynamics," Phys. Rev. E, 1995.
* P. Trautman, A. Krause, "Unfreezing the robot," IROS 2010.
* C. Chen et al., "Crowd-robot interaction: crowd-aware robot navigation with
  attention-based deep RL," ICRA 2019 (CrowdNav).
* C. Mavrogiannis et al., "Core challenges of social robot navigation: a survey," ACM THRI, 2023.
* A. Francis et al., "Principles and guidelines for evaluating social robot navigation
  algorithms," ACM THRI, 2025.
* The MSc thesis that motivated this kit (NCKU) and arXiv:2608.21411.
