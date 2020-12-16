import stable_baselines3

# Set up fake display; otherwise rendering will fail
import os
import gym
# Stable baselines 3
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EveryNTimesteps, \
    EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes

import base64
from pathlib import Path
from torch import nn as nn
import gym
import numpy as np
import base64
import IPython
import PIL.Image
import pyvirtualdisplay
import os

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# Load the agent


env_id = "BipedalWalkerHardcore-v3"
env = gym.make(env_id)
# eval_env = gym.make(env_id)
# time_steps = 100000000
# reward_threshold = 300
# episodes_threshold = 30000
video_folder = './videos'
video_length = 3000
logs_base_dir = "./log"
log_dir = "./log"
# savecounter = 0
# env = gym.make('Pendulum-v0')

model = SAC(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("sac_pendulum")

del model  # remove to demonstrate saving and loading

model = SAC.load("yes")


#
# ### Set log dir
# os.makedirs(logs_base_dir, exist_ok=True)
# ### Enviorment
# env = gym.make(env_id)
# env = Monitor(env, logs_base_dir)
# eval_env = gym.make(env_id)
# score = 0
# log_interval = 10          # Print avg reward after interval


def record_video(env_id, model, video_length=5000, prefix='', video_folder='./'):
    """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)

    # Close the video recorder
    env.close()

    ## Display video
    # Close the video recorder
    eval_env.close()


record_video('BipedalWalkerHardcore-v3', model, video_length=3000, prefix='ppo-cartpole')

# model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)
