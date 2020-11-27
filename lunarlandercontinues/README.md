[![Nextgrid Artificial Intelligence](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/big-banner.jpg)](https://nextgrid.ai)

# LunarLanderContinues-v2

Mission: Take top position in all OpenAI leaderboards. By [Nextgrid](https://nextgrid.ai) and [Deep Learning Labs](https://nextgrid.ai/deep-learning-labs/) find all attempts [here](https://github.com/nextgrid/deep-learning-labs-openAI)

## Solution


### Hyper parameter tuning
HP tuning with Optuna performed on PPO where PPO delivered the best performance. 
See [tune.py](tune.py) for complete HP code.   
`Stables-baselines3`   
`Optuna`  
`Mysql`  
`Docker`  
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
To run training & test checkout this [colab notebook](https://colab.research.google.com/drive/1PN_wl8hcFLTMuD1AXi886eCKjy2lPado?usp=sharing)
**For some reason outputs don't want to be saved despite correct settings in Colab, Result may variate**

### Video Trained model

[LunarLander-v2 Video](https://youtu.be/yhj-t5V9TkY)

## Training
```
Episodes: 166
Num steps: 81000
Mean reward: 148.00 
=========== NEXTGRID.AI ================
Episodes: 167
Num steps: 82000
Mean reward: 133.00 
=========== NEXTGRID.AI ================
Episodes: 168
Num steps: 83000
Mean reward: 131.00 
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
Mean score = 239.5910827084456
```

### Contributors
- [Mathias Åsberg]() 

[![Deep Learning Labs AI ](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/small-banner.jpg)](https://nextgrid.ai/dll)

## Nextgrid AI Seed program

Applications for [Nextgrid](https://nextgrid.ai) AI seed program is now open. We are looking for briliant business & people that are building AI-first solutions. Read more and apply at [AI seed program](https://nextgrid.ai/seed/)

[![Nextgrid Partners](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/partner-banner.jpg)](https://nextgrid.ai/partners/)
