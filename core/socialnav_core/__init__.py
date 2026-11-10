"""socialnav_core: a minimal, self-contained toolkit for crowd-navigation research.

This package is the shared core of the NRSL social-navigation handover. It is a
teaching baseline written from scratch: a Social Force crowd, a unicycle robot,
simple reference controllers, a constant-velocity predictor, standard metrics,
and 2-D visualisation. It is not the method of the original thesis.

Modules
-------
scenarios   four benchmark layouts (corridor, crossing, chokepoint, dense_social)
sfm         Social Force Model used to move the simulated pedestrians
robot       unicycle robot with speed and acceleration limits
prediction  pedestrian predictors (constant velocity; extend with your own)
controllers reference robot controllers and the plug-in interface for yours
observation fixed-size observation vector for learning-based policies
metrics     success / collision / timeout and comfort-related measures
sim         episode loop that ties everything together
env         minimal reset/step environment wrapper for reinforcement learning
viz         static plots and GIF animations of episodes
"""

__version__ = "0.1.0"

from .scenarios import Scenario, Wall, PedSpec, make_scenario, SCENARIOS  # noqa: F401
from .robot import Robot, RobotLimits  # noqa: F401
from .controllers import (  # noqa: F401
    Controller, GoalSeeker, SocialForceController, PredictiveSocialForce,
    LearnedPolicy, make_controller, CONTROLLERS)
from .policy_io import load_policy  # noqa: F401
from .prediction import Predictor, ConstantVelocityPredictor  # noqa: F401
from .sim import Simulator, EpisodeLog, run_episode  # noqa: F401
from .metrics import episode_metrics  # noqa: F401
