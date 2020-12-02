import os
import gym
import numpy as np
import base64
import IPython
import PIL.Image
import pyvirtualdisplay


# Video
from pathlib import Path
from IPython import display as ipythondisplay

# Stable baselines
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
# from stable_baselines3 import TD3
# from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# Values
env_id = 'MountainCar-v0'
video_folder = './videos'
video_length = 3000
logs_base_dir = './runs'  # Log DIR
steps_total = 0  # Keep track of total steps


# Enviorment
env = DummyVecEnv([lambda: gym.make(env_id)])
obs = env.reset()
score = 0
log_interval = 10          # Print avg reward after interval


# Hyperparameters


model = DQN(
    MlpPolicy,
    env,
    # display output when training, 0 = no output, 1 = show output
    verbose=1,
    # gamma=0.99,                        # discount for future rewards
    learning_rate=0.003,                # the learning rate
    # buffer_size=100000,                # size of the replay buffer
    # batch_size=1000,                   # number of transitions sampled from replay buffer
    learning_starts=1000,               # steps before starting training
    # train_freq=1000,                   # update the model every train_freq steps.
    # gradient_steps=1000,               # how many gradient update after each step
    # tau=0.005,                         # the soft update coefficient (“polyak update” of the target networks, between 0 and 1)
    # policy_delay=2,                    # policy and target networks will only be updated once every policy_delay steps per training steps. The Q values will be updated policy_delay more often (update every training step).
    # action_noise=action_noise,         # action noise type. Cf DDPG for the different action noise type.
    # target_policy_noise=0.2,           # standard deviation of Gaussian noise added to target policy
    # target_noise_clip=0.5,             # limit for absolute value of target policy smoothing noise.
    # random_exploration=0.0,            # probability of taking a random action
    tensorboard_log=logs_base_dir,      # tensorboard log dir
)


# Record & Display Video

os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

# Record video


def record_model(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = VecVideoRecorder(env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)
    eval_env.close()


# Display video
def display_videos(video_path='', prefix=''):
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def train_model(name, steps=10000, prefix=env_id, eval=1000):
    model.learn(total_timesteps=steps, log_interval=log_interval)
    model.save(name + "-" + prefix)


def record_video(name, length=1500):
    record_model(env_id, model, video_length=length, prefix=name)
    display_videos('videos', prefix=name)
    print(name, " steps total")


# Training function
def run_training(steps_per_round=0, limit=0):

    global score
    global steps_total

    while score < limit:
        steps_total = steps_total + steps_per_round
        train_model(str(steps_total), steps=steps_per_round)
        new_evaluation = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                                         render=False, callback=None, reward_threshold=None, return_episode_rewards=False)
        score = new_evaluation[0]
        # uncomment to show video from each round
        # record_video(name=steps_total, length=1000)
        print("Mean reward:", score)

    # Threshold reached > evaluate over 100 episodes > Video rec/display
    print("Reward limit achived, messuring over 100ep & recording video, please wait...")
    record_video(name=steps_total, length=1750)
    ep100 = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True,
                            render=False, callback=None, reward_threshold=None, return_episode_rewards=True)
    print("Mean Reward 100 Epispodes: ", ep100[0])


run_training(steps_per_round=20000, limit=1) python train_continuous.py
