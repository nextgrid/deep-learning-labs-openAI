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

hp = {'activation_fn': 'leaky_relu', 'batch_size': 8, 'clip_range': 0.4, 'ent_coef': 1.11811e-07, 'gae_lambda': 0.9,
      'gamma': 0.9999, 'log_std_init': -0.647632, 'lr': 0.000522198, 'max_grad_norm': 0.6, 'net_arch': 'medium',
      'n_epochs': 10, 'n_steps': 2048, 'ortho_init': True, 'sde_sample_freq': 64, 'vf_coef': 0.887769}

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
Episodes: 119
Num steps: 46000
Mean reward: 103.00 
=========== NEXTGRID.AI ================
Episodes: 122
Num steps: 47000
Mean reward: 131.00 
=========== NEXTGRID.AI ================
Episodes: 124
Num steps: 48000
Mean reward: 130.00 
=========== NEXTGRID.AI ================
Episodes: 127
Num steps: 49000
Mean reward: 155.00 
=========== NEXTGRID.AI ================
Episodes: 128
Num steps: 50000
Mean reward: 154.00 
=========== NEXTGRID.AI ================
Episodes: 130
Num steps: 51000
Mean reward: 191.00 
=========== NEXTGRID.AI ================
Episodes: 133
Num steps: 52000
Mean reward: 215.00 
=========== NEXTGRID.AI ================
```
## Evaluation
```buildoutcfg
Score over 100 episodes [230.87654461320452, 101.00565701470434, 225.77502477618404, 225.380276804863, 250.5913839767576, 232.4828750406878, 136.12394859617592, 213.30933859592125, 215.55012852130503, 226.93100289049704, 139.46166704609328, 129.37648131203852, 240.4561317450763, 195.1797969078489, 225.9950710876076, 220.28100618626718, 188.93586771212597, 215.1070461543149, 221.52930006451706, 236.90610536472073, 207.64000898331506, 261.07105403329365, 241.79719062390336, 230.70597684947023, 235.47168117021474, 198.9504316689671, 214.83557671783328, 238.72691597858395, 211.0179616541351, 172.5834095561325, 212.83904000224496, 159.5917435164231, 244.61313756616067, -43.379137275799394, 210.8289258687957, 192.9865229417813, 156.58424819238303, 204.75742451757952, 213.0616139222284, 246.38726033677386, 162.88141265271364, 189.10096507268653, 181.02376344416325, 205.65373105656843, 194.52292777402877, 233.69320723662514, 216.6033894600436, 253.1182740868625, 215.22523315264158, 236.97618628018824, 214.8471364087149, 233.29660176763065, 181.9739557086173, 194.61426260543485, 183.1140096544243, 202.21384410006166, 119.21306671124574, 156.43405787253, 168.47446081819348, 195.60192667730217, 203.40238582828744, 219.26159895474774, 178.24715958013815, 215.85451149189555, 217.75032192212842, 192.91361851726612, 175.6543302552843, 170.29757726678733, 178.14456692764185, 211.19489227218543, 151.73594452320629, 266.52529888923607, 207.65067754698737, 234.6081916675961, 163.45160657785942, 177.87059806240148, 220.90969923593653, 231.4862023711318, 218.511234850914, 204.94029026045007, 188.84094548012996, 210.18160186415884, 248.9994798972294, 239.38595722444364, 108.62183990606292, 209.50716761298415, 213.55796767890988, 242.51657745691912, 153.53375095559522, 151.07531773667563, 248.0695783601542, 253.54871102798612, 201.58898784047688, 234.86784442631281, 262.306437472923, 227.36541440269067, 220.03310719697748, 173.82906423909566, 237.4024357660282, 224.44493214756423]
MEAN: 203.1099587947141
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
