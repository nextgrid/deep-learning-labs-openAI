from torch import nn as nn
import gym
import numpy as np
import base64
import IPython
import PIL.Image
import pyvirtualdisplay
import os
import optuna

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Stable baselines 3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EveryNTimesteps, \
    EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes

# ======================================================================== Enviorment settings

env_id = 'LunarLander-v2'
env = gym.make(env_id)

time_steps = 2000000
reward_threshold = -150
episodes_threshold = 1000
episodes = 0
mean_reward = 0
# eval_env = DummyVecEnv([lambda: gym.make(env_id)])
video_folder = './videos'
video_length = 3000
logs_base_dir = "./log"
log_dir = "./log"
os.makedirs(logs_base_dir, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)
env = Monitor(env, log_dir)
eval_env = DummyVecEnv([lambda: gym.make(env_id)])

obs = env.reset()



hp = {'batch_size': 64, 'n_steps': 2048, 'gamma': 0.9999, 'lr': 0.00045431178185513705,
      'ent_coef': 0.0007188057028259272, 'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.9, 'max_grad_norm': 1,
      'vf_coef': 0.49782864770311314, 'net_arch': 'medium', 'log_std_init': -0.9591570156697942, 'sde_sample_freq': 32,
      'ortho_init': False, 'activation_fn': 'leaky_relu'}

model = PPO(
    MlpPolicy,
    env,
    n_steps=hp["n_steps"],
    batch_size=hp["batch_size"],
    gamma=hp["gamma"],
    learning_rate=hp["lr"],
    ent_coef=hp["ent_coef"],
    clip_range=hp["clip_range"],
    n_epochs=hp["n_epochs"],
    gae_lambda=hp["gae_lambda"],
    max_grad_norm=hp["max_grad_norm"],
    vf_coef=hp["vf_coef"],
    sde_sample_freq=hp["sde_sample_freq"],
    policy_kwargs=dict(
        log_std_init=hp["log_std_init"],
        net_arch=[dict(pi=[128, 128], vf=[128, 128])],
        activation_fn=nn.LeakyReLU,
        ortho_init=hp["ortho_init"],

    ),
    verbose=0
)

# ======================================================================== Hyper Parameters

# ======================================================================== Evaluation

import os


# Record video
def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """

    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


class RewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                global episodes
                global mean_reward
                episodes = len(y)
                # print(episodes)
                mean_reward = np.mean(y[-50:])
                mean_reward = round(mean_reward, 0)
                if self.verbose > 0:
                    print(f"Episodes: {episodes}")
                    print(f"Num steps: {self.num_timesteps}")
                    print(f"Mean reward: {mean_reward:.2f} ")
                    print("=========== NEXTGRID.AI ================")
                # Report intermediate objective value to Optima and Handle pruning
                # trial.report(episodes, self.num_timesteps)
                # if trial.should_prune():
                #     raise optuna.TrialPruned()

                # New best model, you could save the agent here
                if episodes > episodes_threshold:
                    print("Reward threshold achieved")
                    return False

                # New best model, you could save the agent here
                if mean_reward > reward_threshold:
                    new_evaluation = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True, render=False,
                                                     callback=None, reward_threshold=None, return_episode_rewards=False)
                    score = new_evaluation[0]

                    print("<======SCORE======>")
                    print(score)
                    if score > reward_threshold:

                        print("Woop")
                        return False

        return True


# ======================================================================== Training
print("hello world")
callback = RewardCallback(check_freq=10000, log_dir=log_dir)
model.learn(total_timesteps=int(time_steps), callback=callback)
record_video(env_id, model, video_length=video_length, prefix="LunarLander-v2-PPO-")
# Record the video starting at the first step


# ==== Rest environment
# del model
# env.reset()
