# NRSL social-navigation handover: one command per step.
#   make setup       install the Python dependencies into .venv
#   make doctor      check what this machine can run (changes nothing)
#   make test        unit tests of the core library
#   make sim2d       stage 1: 2-D benchmark of the baselines (CSV, tables, figures)
#   make gifs        stage 1: animations
#   make train       RL: train PPO with stable-baselines3 (RUN=ppo_mixed, STEPS=1500000)
#   make train-quick RL: 200k-step run to check the pipeline (a few minutes)
#   make play        RL: evaluate and animate the trained policy in 2-D
#   make export      RL: re-export a model to policy.pt / policy.npz / policy_meta.json
#   make mujoco      MuJoCo: run a controller or the trained policy (GIFs + metrics)
#   make worlds      stage 2: Gazebo worlds from the scenarios
#   make ros         stage 2+3: build the ROS 2 packages
#   make gazebo      stage 2: Gazebo with GUI (SCENARIO, CONTROLLER, POLICY)
#   make smoke       stage 2: headless end-to-end Gazebo test
#   make slides      rebuild the handover slides
#   make all         test + sim2d + gifs + worlds + ros + slides
HANDOVER := $(abspath .)
PY := $(if $(wildcard .venv/bin/python),$(HANDOVER)/.venv/bin/python,python3)
export PYTHONPATH := $(HANDOVER)/core:$(PYTHONPATH)
SCENARIO ?= corridor
CONTROLLER ?= predictive_sf
RUN ?= ppo_mixed
STEPS ?= 1500000
POLICY ?= $(HANDOVER)/stage1_python_sim/rl/runs/$(RUN)/policy.pt

.PHONY: all setup doctor test sim2d gifs train train-quick play export mujoco worlds ros gazebo smoke slides clean
all: test sim2d gifs worlds ros slides

setup:
	bash scripts/setup_python.sh

doctor:
	$(PY) scripts/doctor.py

test:
	$(PY) -m pytest core/tests -q

sim2d:
	$(PY) stage1_python_sim/run_benchmark.py --seeds 20

gifs:
	$(PY) stage1_python_sim/animate.py --all

train:
	$(PY) stage1_python_sim/rl/train_sb3.py --run-name $(RUN) --timesteps $(STEPS)

train-quick:
	$(PY) stage1_python_sim/rl/train_sb3.py --run-name quick --timesteps 200000 --eval-every 50000

play:
	$(PY) stage1_python_sim/rl/play.py --policy $(POLICY) --episodes 20 --gif

export:
	$(PY) stage1_python_sim/rl/export_policy.py stage1_python_sim/rl/runs/$(RUN)/best_model.zip

mujoco:
	$(PY) stage2_mujoco/mujoco_crowd.py --all --gif stage2_mujoco/outputs/mujoco.gif \
	    $(if $(wildcard $(POLICY)),--policy $(POLICY),--controller $(CONTROLLER))

worlds:
	$(PY) stage2_gazebo/crowd_gazebo/crowd_gazebo/generate_worlds.py --out stage2_gazebo/crowd_gazebo/worlds

ros: worlds
	bash scripts/build_ros.sh

gazebo:
	bash -c "source scripts/sim_env.sh && ros2 launch crowd_gazebo gazebo_crowd.launch.py scenario:=$(SCENARIO) controller:=$(CONTROLLER) $(if $(filter learned,$(CONTROLLER)),policy:=$(POLICY),)"

smoke:
	bash scripts/gazebo_smoke_test.sh $(SCENARIO) $(CONTROLLER) 90 $(if $(filter learned,$(CONTROLLER)),$(POLICY),)

slides:
	$(PY) slides/build_slides.py

clean:
	rm -rf ros_ws/build ros_ws/install ros_ws/log stage1_python_sim/outputs stage2_mujoco/outputs
