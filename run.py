import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# hyper parameters
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 128  # NN hidden layer size
BATCH_SIZE = 32  # Q-learning batch size

# %% DQN NETWORK ARCHITECTURE


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


model = Network()
optimizer = optim.Adam(model.parameters(), LR)

# %% SELECT ACTION USING GREEDY ALGORITHM
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # return argmaxQ
        return model(Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        # return random action
        return torch.LongTensor([[random.randrange(2)]])

# %% MEMORY REPLAY


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# %%
memory = ReplayMemory(10000)
episode_durations = []


def run_episode(e, environment):
    state = environment.reset()
    steps = 0
    while True:
        environment.render()
        action = select_action(torch.FloatTensor([state]))
        next_state, reward, done, _ = environment.step(action.numpy()[0, 0])

        # negative reward when attempt ends
        if done:
            reward = -10

        memory.push((torch.FloatTensor([state]),
                     action,  # action is already a tensor
                     torch.FloatTensor([next_state]),
                     torch.FloatTensor([reward])))

        learn()

        state = next_state
        steps += 1

        if done:
            #print("{2} Episode {0} finished after {1} steps".format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            print("Episode {0} finished after {1} steps".format(e, steps))
            episode_durations.append(steps)
            break

# %% TRAIN THE MODEL


def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(
        *transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values.reshape_as(
        expected_q_values), expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% RUN AND SHOW THE RESULT


EPISODES = 100  # number of episodes
# establish the environment
env = gym.make('CartPole-v0')

for e in range(EPISODES):
    run_episode(e, env)

print('Complete')
plt.plot(episode_durations)
plt.show()

# %%
