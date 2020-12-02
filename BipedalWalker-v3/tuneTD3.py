from torch import nn as nn
import gym
import numpy as np
import base64
import IPython
import PIL.Image
import pyvirtualdisplay
import os
import optuna

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
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

# ======================================================================== Environment settings

env_id = 'BipedalWalker-v3'
# env_id = 'CartPole-v1'

timesteps = 2000000
reward_threshold = 200
episodes_threshold = 1000
study_name = "BipedalWalker"
eval_env = gym.make(env_id)
video_folder = './videos'
video_length = 3000
logs_base_dir = "./log"
log_dir = "./log"


# ======================================================================== Optuna Loop
def objective(trial):
    # gym environment & variables

    env = gym.make(env_id)
    # Parallel environments
    # env = make_vec_env(gym.make(env_id), n_envs=4)
    os.makedirs(logs_base_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    global episodes
    global mean_reward
    episodes = 0
    mean_reward = 0

    # ======================================================================== Hyper Parameters
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 2e-4, 6e-4)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])

    episodic = trial.suggest_categorical("episodic", [True, False])

    if episodic:
        n_episodes_rollout = 1
        train_freq, gradient_steps = -1, -1
    else:
        train_freq = trial.suggest_categorical("train_freq", [1, 16, 128, 256, 1000, 2000])
        gradient_steps = train_freq
        n_episodes_rollout = -1

    # noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    # if noise_type == "normal":
    #     noise_type = NormalActionNoise(
    #         mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
    #     )
    # elif noise_type == "ornstein-uhlenbeck":
    #     noise_type = OrnsteinUhlenbeckActionNoise(
    #         mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
    #     )

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
    }[net_arch]

    model = TD3(
        MlpPolicy,
        env,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        train_freq=train_freq,
        learning_rate=learning_rate,
        gradient_steps=gradient_steps,
        n_episodes_rollout=n_episodes_rollout,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=0
    )

    # ======================================================================== Hyper Parameters

    # ======================================================================== Evaluation

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
                    mean_reward = np.mean(y[-30:])
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
                        print("REWARD ACHIVED")
                        return False

                    # New best model, you could save the agent here
                    if mean_reward > reward_threshold:
                        print("REWARD ACHIVED")
                        return False

            return True

    # ======================================================================== Training

    callback = RewardCallback(check_freq=2500, log_dir=log_dir)
    model.learn(total_timesteps=int(timesteps), callback=callback)

    # ==== Rest environment
    del model
    env.reset()

    return episodes


storage = 'mysql://root:@34.122.181.208/rl'

study = optuna.create_study(study_name=study_name, storage=storage,
                            pruner=optuna.pruners.MedianPruner(), load_if_exists=True)
study.optimize(objective, n_trials=10, n_jobs=1)
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
print(df) , direction='maximize'
print(study.best_params) # Get best params
print(study.best_value)  # Get best objective value.
print(study.best_trial)  # Get best trial's information.
print(study.trials)  # Get all trials' information.
len(study.trials)  # Get number of trails.

