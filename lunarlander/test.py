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

# Enviorment
env_id = 'LunarLander-v2'
env = gym.make(env_id)
eval_env = gym.make(env_id)

# Video
video_folder = './videos'
video_length = 3000

# Logs
logs_base_dir = "./log"
os.makedirs(logs_base_dir, exist_ok=True)

# Vars
score = 0
steps_total = 0
log_interval = 100


global model
global n_total_timesteps
n_total_timesteps = 1
obs = env.reset()


model = DQN(
    MlpPolicy,
    env,
    gamma=0.95,
    learning_rate=0.0004742978025806393,
    batch_size=128,  # batch_size,
    buffer_size=50000,  # buffer_size,
    train_freq=4,  # train_freq,
    gradient_steps=-1,  # gradient_steps,
    #         n_episodes_rollout=n_episodes_rollout,
    exploration_fraction=0.12,  # exploration_fraction,
    exploration_final_eps=0.1,  # exploration_final_eps,
    #         target_update_interval=target_update_interval,
    learning_starts=1000,
    policy_kwargs=dict(net_arch=[64, 64]),
    #         tensorboard_log=logs_base_dir,
    verbose=1
)


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        print(self.num_timesteps)
        global n_total_timesteps
        n_total_timesteps = self.num_timesteps
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=-200, verbose=1)
eval_callback = EvalCallback(
    eval_env, callback_on_new_best=callback_on_best, verbose=1)


# Traing
# Create the callback list
callbacks = CallbackList([CustomCallback(1), eval_callback])


model.learn(5000, callback=callbacks, log_interval=4, eval_env=None, eval_freq=10,
            n_eval_episodes=5)
print(n_total_timesteps)

# ep100 = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
#                         render=False, callback=None, reward_threshold=None, return_episode_rewards=True)
# print("Mean reward 100 ep: ", ep100[0])
