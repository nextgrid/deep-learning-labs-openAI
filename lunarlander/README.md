[![Nextgrid Artificial Intelligence](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/big-banner.jpg)](https://nextgrid.ai)

# LunarLander-v2

Mission: Take top position in all OpenAI leaderboards. By [Nextgrid](https://nextgrid.ai) and [Deep Learning Labs](https://nextgrid.ai/deep-learning-labs/) find all attempts [here](https://github.com/nextgrid/deep-learning-labs-openAI)

## Solution


### Hyper parameter tuning
HP tuning performed on DQN & PPO where PPO delivered the best performance. 
See [tune.py](tune.py) for complete HP code.   
`Stables-baselines3`   
`Optuna`  
`Mysql`
`Kubernetes` 

Optima was used to find the best hyper parameter tuning  
watch the [video](https://www.youtube.com/watch?v=a0oA5VmVFhQ&feature=youtu.be)

```
### Hyperparameters 

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
```

## Training
```
=========== NEXTGRID.AI ================
Episodes: 314
Num steps: 111000
Mean reward: 169.00 
=========== NEXTGRID.AI ================
Episodes: 322
Num steps: 114000
Mean reward: 180.00 
=========== NEXTGRID.AI ================
Episodes: 333
Num steps: 117000
Mean reward: 190.00 
=========== NEXTGRID.AI ================
Episodes: 342
Num steps: 120000
Mean reward: 193.00 
=========== NEXTGRID.AI ================
Episodes: 352
Num steps: 123000
Mean reward: 195.00 
=========== NEXTGRID.AI ================
Episodes: 362
Num steps: 126000
Mean reward: 212.00 
=========== NEXTGRID.AI ================
```
## Evaluation
```buildoutcfg
Score over 100 episodes 225.79136232746092
```

### Stack

```
Stable-baselines3
Pytorch
Gym
Optima
Docker
Kubernetes
```

### How to run

To run hyper parameter optimization run the file `tune.py`  
To run training & test checkout this [colab notebook](https://colab.research.google.com/drive/1bhJw-oXYbLApc3EpfyX3IZCn6_S1O_Id#scrollTo=U7D2_Fjzf935)

### Video Trained model

[LunarLander-v2 Video](https://www.youtube.com/watch?v=OKQFbvNj6JI&feature=youtu.be)

## Training
```
=========== NEXTGRID.AI ================
Episodes: 314
Num steps: 111000
Mean reward: 169.00 
=========== NEXTGRID.AI ================
Episodes: 322
Num steps: 114000
Mean reward: 180.00 
=========== NEXTGRID.AI ================
Episodes: 333
Num steps: 117000
Mean reward: 190.00 
=========== NEXTGRID.AI ================
Episodes: 342
Num steps: 120000
Mean reward: 193.00 
=========== NEXTGRID.AI ================
Episodes: 352
Num steps: 123000
Mean reward: 195.00 
=========== NEXTGRID.AI ================
Episodes: 362
Num steps: 126000
Mean reward: 212.00 
=========== NEXTGRID.AI ================
```
## Evaluation
```buildoutcfg
Score over 100 episodes 225.79136232746092
```

### Contributors
- [Mathias Åsberg]() 

[![Deep Learning Labs AI ](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/small-banner.jpg)](https://nextgrid.ai/dll)

## Nextgrid AI Seed program

Applications for [Nextgrid](https://nextgrid.ai) AI seed program is now open. We are looking for briliant business & people that are building AI-first solutions. Read more and apply at [AI seed program](https://nextgrid.ai/seed/)

[![Nextgrid Partners](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/partner-banner.jpg)](https://nextgrid.ai/partners/)
