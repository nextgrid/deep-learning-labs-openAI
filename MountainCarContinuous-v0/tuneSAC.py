from torch import nn as nn
import gym
import numpy as np
import base64
import IPython
import PIL.Image
import pyvirtualdisplay
import os
import optuna
from typing import Any, Dict
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# Stable baselines 3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EveryNTimesteps, \
    EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes

# ======================================================================== Environment settings
env_id = 'MountainCarContinuous-v0'
# env_id = 'CartPole-v1'
timesteps = 2000000
reward_threshold = 90
episodes_threshold = 300
callback_check_freq = 2000
study_name = "MountainCarContinuous_Trail_4"
eval_env = gym.make(env_id)
video_folder = './videos'
video_length = 3000
logs_base_dir = "./log"
log_dir = "./log"


def sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 9e-4, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(25e3), int(5e4), int(1e5), int(3e5)])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 10, 50, 100])
    train_freq = trial.suggest_categorical("train_freq", [8, 16, 32, 64, 128])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02])
    gradient_steps = train_freq
    ent_coef = "auto"
    sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128])
    use_sde = True
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = {
        "small": [64, 64],
        "medium": [128, 128],
        "big": [256, 256],
    }[net_arch]
    target_entropy = "auto"
    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "sde_sample_freq": sde_sample_freq,
        "use_sde": use_sde,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }


# ======================================================================== Optuna Loop
def objective(trial):
    # ======================================================================== Configure environment
    os.makedirs(logs_base_dir, exist_ok=True)
    env = gym.make(env_id)
    env = Monitor(env, log_dir)
    global episodes
    global mean_reward
    episodes = 0
    mean_reward = 0

    # ======================================================================== HyperParameters
    hp = sac_params(trial)
    print(hp)


    model = SAC(
        MlpPolicy,
        env,
        gamma=hp["gamma"],
        learning_rate=hp["learning_rate"],
        batch_size=hp["batch_size"],
        buffer_size=hp["buffer_size"],
        learning_starts=hp["learning_starts"],
        train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"],
        ent_coef=hp["ent_coef"],
        tau=hp["tau"],
        use_sde=hp["use_sde"],
        sde_sample_freq=hp["sde_sample_freq"],
        target_entropy=hp["target_entropy"],
        policy_kwargs=hp["policy_kwargs"],
        tensorboard_log=logs_base_dir,
        verbose=0
    )

    def evaluate(model, env):
        """
        :param reward_threshold: (int)
        :param model (the model)

        """
        evals = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True, render=False, callback=None,
                                reward_threshold=None, return_episode_rewards=True)
        evals_mean = np.mean(evals[0])
        print("Score over 100 episodes", evals[0])
        print(evals_mean)

        return evals_mean

        # ======================================================================== Custom Callback - Evaluation

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
                    mean_reward = np.mean(y[-100:])
                    mean_reward = round(mean_reward, 0)
                    if self.verbose > 0:
                        print(f"Episodes: {episodes}")
                        print(f"Num steps: {self.num_timesteps}")
                        print(f"Mean Episode reward: {mean_reward:.2f} ")
                        print(f"Last Episode reward: {y[-1]:.2f} ")
                        print("=========== NEXTGRID.AI ================")
                    # Report intermediate objective value to Optima and Handle pruning
                    trial.report(mean_reward, episodes)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    # New best model, you could save the agent here
                    if episodes > episodes_threshold:
                        print("Episode threshold")
                        print("Aborting training")
                        return False

                    # New best model, you could save the agent here
                    if mean_reward >= reward_threshold:
                        print("Reward threshold achieved")
                        print("Evaluating model....")
                        evals = evaluate(model, eval_env)
                        print(evals)
                        print(f"Evaluation over 100 Episodes: {evals:.2f} ")
                        if evals >= reward_threshold:
                            print(f"MISSION COMPLETED")
                            print(f"Score: {evals:.2f} reached at Episode: {episodes} ")
                            return False

            return True

    # ======================================================================== Training
    callback = RewardCallback(check_freq=callback_check_freq, log_dir=log_dir)
    model.learn(total_timesteps=int(timesteps), callback=callback)
    # ==== Rest environment
    del model
    env.reset()
    return episodes


storage = 'mysql://root:@34.122.181.208/rl'
study = optuna.create_study(study_name=study_name, storage=storage, pruner=optuna.pruners.MedianPruner(), load_if_exists=True,)
study.optimize(objective, n_trials=3, n_jobs=1)

# study = optuna.create_study(study_name=study_name, storage=storage,
#                             pruner=optuna.pruners.MedianPruner(), , direction='maximize', load_if_exists=True)
# df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
# print(df)
print(study.best_params)  # Get best params
print(study.best_value)  # Get best objective value.
print(study.best_trial)  # Get best trial's information.
# print(study.trials)  # Get all trials' information.
# len(study.trials)  # Get number of trails.
