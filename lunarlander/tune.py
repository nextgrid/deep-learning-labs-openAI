from torch import nn as nn
import gym
import numpy as np
import base64
import IPython
import PIL.Image
import pyvirtualdisplay
import os
import optuna

# Video
from pathlib import Path
from IPython import display as ipythondisplay

# Stable baselines

# DQN specific

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
timesteps = 700000
reward_threshold = 200
study_name = "lunarlanderPP0_100"
eval_env = gym.make(env_id)
video_folder = './videos'
video_length = 3000
logs_base_dir = "./log"
log_dir = "./log"


# ======================================================================== Optuna Loop
def objective(trial):
    # gym environment & variables

    env = gym.make(env_id)
    os.makedirs(logs_base_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    global episodes
    episodes = 0

    # print out observation space
    # print('State shape: ', env.observation_space.shape)

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    lr_schedule = "constant"

    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    sde_sample_freq = trial.suggest_categorical(
        "sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    ortho_init = False
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    # activation_fn = trial.suggest_categorical(
    #     "activation_fn", ["tanh", "relu"])

    #     # TODO: account when using multiple envs
    # if batch_size > n_steps:
    #     batch_size = n_steps

    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU,
                     "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    model = PPO(
        MlpPolicy,
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        sde_sample_freq=sde_sample_freq,
        policy_kwargs=dict(
            log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,

        ),
        verbose=0
    )


    #       n_envs: 16
    #   n_timesteps: !!float 1e6
    #   policy: 'MlpPolicy'
    #   n_steps: 1024
    #   batch_size: 64
    #   gae_lambda: 0.98
    #   gamma: 0.999
    #   n_epochs: 4
    #   ent_coef: 0.01

    #     gamma = trial.suggest_categorical(
    #         "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    #     learning_rate = trial.suggest_loguniform("lr", 4.3e-4, 7.3e-4)
    #     batch_size = trial.suggest_categorical(
    #         "batch_size", [16, 32, 64, 100, 128, 256, 512])
    #     buffer_size = trial.suggest_categorical(
    #         "buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    # #     exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    # #     exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    #     target_update_interval = trial.suggest_categorical(
    #         "target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    #     learning_starts = trial.suggest_categorical(
    #         "learning_starts", [0, 1000, 5000, 10000])
    #     train_freq = trial.suggest_categorical(
    #         "train_freq", [1, 4, 8, 16, 128, 256, 1000])
    # #     subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    # #     gradient_steps = max(train_freq // subsample_steps, 1)
    # #     n_episodes_rollout = -1
    #     net_arch = trial.suggest_categorical(
    #         "net_arch", ["tiny", "small", "medium"])
    #     net_arch = {"tiny": [64], "small": [
    #         64, 64], "medium": [256, 256]}[net_arch]

    # ======================================================================== Hyper Parameters

    # ======================================================================== Evaluation


    class RewardCallback(BaseCallback):
        global episodes
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
                    # Mean training reward over the last 100 episodes
                    # print(y)
                    global episodes
                    episodes = len(y)
                    print(episodes)
                    mean_reward = np.mean(y[-100:])
                    mean_reward = round(mean_reward, 0)
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Mean reward: {mean_reward:.2f} ")
                        # print(
                        #     f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    # New best model, you could save the agent here
                    if mean_reward > reward_threshold:
                        print("REWARD ACHIVED")
                        return False

            return True


# ======================================================================== Training

    callback = RewardCallback(check_freq=5000, log_dir=log_dir)
    model.learn(total_timesteps=int(timesteps), callback=callback)
    print(episodes)

    # ==== Rest environment
    del model
    env.reset()

    return episodes

storage = optuna.storages.RedisStorage(
    url='redis://34.123.159.224:6379/DB1',
)

study = optuna.create_study(
    study_name=study_name, storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=10)
print(study.best_params)

# direction='maximize'
