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
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Stable baselines 3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EveryNTimesteps, EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes

# ======================================================================== Enviorment settings

env_id = 'MountainCar-v0'
timesteps = 700000
reward_threshold = -110
study_name = "MountainCar3"
eval_env = gym.make(env_id)
video_folder = './videos'
video_length = 3000
logs_base_dir = "./log"
log_dir = "./log"


# ======================================================================== Optuna Loop
def objective(trial):

    # gym enviorment & variables

    env = gym.make(env_id)
    os.makedirs(logs_base_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    global episodes
    episodes = 0

    # print out observation space
    # print('State shape: ', env.observation_space.shape

    global model

    obs = env.reset()

    # ======================================================================== Optuna

    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 4.3e-4, 7.3e-4)
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_uniform(
        "exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform(
        "exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical(
        "target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 5000, 10000])
    train_freq = trial.suggest_categorical(
        "train_freq", [1, 4, 8, 16, 128, 256, 1000])
#     subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
#     gradient_steps = max(train_freq // subsample_steps, 1)
#     n_episodes_rollout = -1
    net_arch = trial.suggest_categorical(
        "net_arch", ["tiny", "small", "medium"])
    net_arch = {"tiny": [64], "small": [
        64, 64], "medium": [256, 256]}[net_arch]

    # ======================================================================== Hyper Parameters

    model = DQN(
        MlpPolicy,
        env,
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        train_freq=train_freq,
        gradient_steps=-1,  # gradient_steps,
        #         n_episodes_rollout=n_episodes_rollout,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        learning_starts=learning_starts,
        policy_kwargs=dict(net_arch=net_arch),
        #         tensorboard_log=logs_base_dir,
        verbose=0
    )

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

    callback = RewardCallback(check_freq=1000, log_dir=log_dir)
    # Train the agent

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
study.optimize(objective, n_trials=30)
print(study.best_params)
print(study.best_params)
