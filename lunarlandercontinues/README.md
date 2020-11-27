[![Nextgrid Artificial Intelligence](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/big-banner.jpg)](https://nextgrid.ai)

# LunarLanderContinues-v2

Mission: Take top position in all OpenAI leaderboards. By [Nextgrid](https://nextgrid.ai) and [Deep Learning Labs](https://nextgrid.ai/deep-learning-labs/) find all attempts [here](https://github.com/nextgrid/deep-learning-labs-openAI)

## Solution


### Hyper parameter tuning
HP tuning with Optuna
See [tunePPO.py](tunePPO.py) for complete HP code.   
`Stables-baselines3`   
`Optuna`  
`Mysql`  
`Docker`  
`Kubernetes` 

Optima / Docker / Kubernetes was used to find the best hyper parameters


```
### Hyperparameters 

hp = {'activation_fn': 'relu', 'batch_size': 8, 'clip_range': 0.4, 
      'ent_coef': 2.89108e-05, 'gae_lambda': 0.92, 'gamma': 0.99, 'log_std_init': -0.00775684,
      'lr': 0.000242873, 'max_grad_norm': 0.3, 'net_arch': 'medium',
      'n_epochs': 10, 'n_steps': 1024, 'ortho_init': True, 'sde_sample_freq': 8, 'vf_coef': 0.856625}

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
Episodes: 169
Num steps: 84000
Mean reward: 147.00 
=========== NEXTGRID.AI ================
Episodes: 170
Num steps: 85000
Mean reward: 150.00 
=========== NEXTGRID.AI ================
Episodes: 172
Num steps: 86000
Mean reward: 160.00 
=========== NEXTGRID.AI ================
Episodes: 174
Num steps: 87000
Mean reward: 174.00 
=========== NEXTGRID.AI ================
Episodes: 176
Num steps: 88000
Mean reward: 190.00 
=========== NEXTGRID.AI ================
Episodes: 178
Num steps: 89000
Mean reward: 210.00 
=========== NEXTGRID.AI ================
```
## Evaluation
```buildoutcfg
Score over 100 episodes = [251.96296540974888, 268.35536734124645, 254.47770373640788, 240.81771828101893, 0.451302015052633, 243.86348009810195, 249.3015898655448, 245.6210808115789, 257.8594839240616, 234.56861385409138, 264.151358569239, 247.85814094198207, 246.43129194671226, 234.7021727330193, 240.8208049779611, 225.24902135639172, 254.03558719495618, 242.71990446081622, 238.41749136596403, 263.82774689321604, 260.0223910544711, 264.0646604355377, 168.868551022762, 257.6135853461569, 224.7403381765052, 272.0986926324243, 280.1336906594461, 265.8432978047108, 244.1279538295671, 241.667759608345, 238.42458440428254, 256.1741388682512, 229.17794523968067, 243.80493187787323, 235.3943282120312, 232.71383400963222, 250.1930714025949, 266.30138927411446, 248.36901857767947, 253.84183459648523, 258.70920075573315, 262.1183082251715, 247.0929964472305, 244.5373628420017, 238.53400307528605, 250.00915081384946, 271.9847480834892, 233.85501541147255, 265.6503016369163, 250.68941951617077, 259.3460534832894, 236.30583934442618, 260.0331376491822, 238.45959578733508, 249.08631844147956, 242.81917058171683, 272.66470838031944, 237.0497720795012, 242.98487230165685, 241.04413548494816, 33.71933577817282, 278.98069980527487, 238.70869245374467, 247.10439227573872, 255.7426885657647, 223.56695266403358, 229.66748803893634, 233.20584060707233, 269.05182466193384, 261.5843141814882, 257.9631264440162, 245.53731171596763, 258.2390466557469, 265.0315664079186, 253.2374070924753, 241.7159009904781, 234.19517936105268, 275.32524924674175, -16.21543050367002, 263.8058713633763, 262.2367431699763, 247.4279292055675, 215.07740973953628, 233.40247673779635, 227.36433735574846, 270.70793441018725, -9.063842056710996, 257.6172886361456, 247.39544219491444, 270.35327873564506, 221.47778804876373, 262.9793931188369, 244.32953679276943, 278.89022896875065, 265.78679395545413, 276.8705572825596, 260.9864645034827, 262.4854435286363, 241.57430818129427, 228.99836737810483]
Mean = 239.5910827084456
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

To run hyper parameter optimization run the file `tunePPO.py`  
To run training & test checkout this [colab notebook](https://colab.research.google.com/drive/1PN_wl8hcFLTMuD1AXi886eCKjy2lPado?usp=sharing)
**For some reason outputs don't want to be saved despite correct settings in Colab, Result may variate**

### Video of tsrained model

[LunarLander-v2 Video](https://youtu.be/yhj-t5V9TkY)


### Contributors
- [Mathias Åsberg]() 

[![Deep Learning Labs AI ](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/small-banner.jpg)](https://nextgrid.ai/dll)

## Nextgrid AI Seed program

Applications for [Nextgrid](https://nextgrid.ai) AI seed program is now open. We are looking for briliant business & people that are building AI-first solutions. Read more and apply at [AI seed program](https://nextgrid.ai/seed/)

[![Nextgrid Partners](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/partner-banner.jpg)](https://nextgrid.ai/partners/)
