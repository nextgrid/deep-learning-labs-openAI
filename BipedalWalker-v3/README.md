[![Nextgrid Artificial Intelligence](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/big-banner.jpg)](https://nextgrid.ai)

# BipedalWalker-v3

Mission: Take top position in all OpenAI leaderboard. By [Nextgrid](https://nextgrid.ai) and [Deep Learning Labs](https://nextgrid.ai/deep-learning-labs/) find all attempts [here](https://github.com/nextgrid/deep-learning-labs-openAI). The BipedalWalker was **solved on 164 Episodes** and is currently the no1 result.

## Solution
After running DQN & TD3 I switched to SAC: Soft Actor Critic (SAC) Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.

### HyperParameter tuning
HP tuning with Optuna
See [tuneSAC.py](tuneSAC.py) for complete HP code.  

**Stack**:  
`Stables-baselines3`   
`Optuna`  
`Mysql`  
`Docker`  
`Kubernetes` 

Optima / Docker / Kubernetes was used to find the best hyper parameters

### Optuna trials
Did run +100 trials where trial 41 delivered the best result solving the BipedalWalker problem on 194 episodes. Did manage to get it down to 164 in colab 

```
FrozenTrial(number=61, value=194.0, datetime_start=datetime.datetime(2020, 12, 9, 5, 14, 55), datetime_complete=datetime.datetime(2020, 12, 9, 5, 58, 7), params={'batch_size': 128, 'buffer_size': 50000, 'gamma': 0.99, 'learning_starts': 1000, 'log_std_init': 0.409723, 'lr': 0.000314854, 'net_arch': 'medium', 'tau': 0.02, 'train_freq': 128}, distributions={'batch_size': CategoricalDistribution(choices=(64, 128, 256, 512)), 'buffer_size': CategoricalDistribution(choices=(50000, 100000, 300000, 500000, 1000000)), 'gamma': CategoricalDistribution(choices=(0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999)), 'learning_starts': CategoricalDistribution(choices=(100, 1000, 10000, 20000)), 'log_std_init': UniformDistribution(high=1, low=-4), 'lr': LogUniformDistribution(high=0.0009, low=0.0002), 'net_arch': CategoricalDistribution(choices=('medium', 'big')), 'tau': CategoricalDistribution(choices=(0.001, 0.005, 0.01, 0.02)), 'train_freq': CategoricalDistribution(choices=(32, 64, 128))}, user_attrs={}, system_attrs={}, intermediate_values={}, trial_id=215, state=TrialState.COMPLETE)

params={'batch_size': 128, 'buffer_size': 50000, 'gamma': 0.99, 'learning_starts': 1000, 'log_std_init': 0.409723, 'lr': 0.000314854, 'net_arch': 'medium', 'tau': 0.02, 'train_freq': 128} 
```


```
### Hyperparameters 

hp = {'batch_size': 128, 'buffer_size': 50000, 'gamma': 0.99, 
      'learning_starts': 1000, 'log_std_init': 0.409723, 
      'lr': 0.000314854, 'net_arch': 'medium', 
      'tau': 0.02, 'ent_coef': 'auto', 'target_entropy': 'auto',
      'train_freq': 128}


print(hp)


model = SAC(
    MlpPolicy,
    env,
    gamma=hp["gamma"],
    learning_rate=hp["lr"],
    batch_size=hp["batch_size"],
    buffer_size=hp["buffer_size"],
    learning_starts=hp["learning_starts"],
    train_freq=hp["train_freq"],
    gradient_steps=hp["train_freq"],
    ent_coef=hp["ent_coef"],
    tau=hp["tau"],
    target_entropy=hp["target_entropy"],
    policy_kwargs=dict(log_std_init=hp["log_std_init"], net_arch=[400, 300]),
    verbose=0
)
```

## Training
```
=========== NEXTGRID.AI ================
Episodes: 156
Num steps: 145000
Mean Episode reward: 219.00 
Last Episode reward: 299.80 
=========== NEXTGRID.AI ================
Episodes: 159
Num steps: 147500
Mean Episode reward: 176.00 
Last Episode reward: -24.58 
=========== NEXTGRID.AI ================
Episodes: 161
Num steps: 150000
Mean Episode reward: 178.00 
Last Episode reward: 306.47 
=========== NEXTGRID.AI ================
Episodes: 164
Num steps: 152500
Mean Episode reward: 306.00 
Last Episode reward: 307.74 
=========== NEXTGRID.AI ================
Reward threshold achieved
Evaluating model....
Score over 100 episodes [308.2105961556759, 308.7545116786518, 307.35891844988515, 306.49873706355817, 306.14999475808986, 304.54651776834976, 306.8431018615281, 306.2894691873911, 306.9236794768736, 307.32380583578816, 306.42066097550156, 308.089091895441, 307.039820877903, 306.55900825224927, 306.85847675759135, 306.80544472745567, 307.1472148625892, 307.545056926034, 307.0899983193522, 305.7578543565915, 307.503701526062, 305.95886609987633, 307.4947148979468, 308.54527560219503, 119.12757970955778, 306.82872944270304, 305.5784539690095, 308.5990520027172, 305.846679143209, 305.6356648901386, 306.19742303794453, 310.4742526098147, 306.97823252326737, 307.89536245050056, 306.681235046599, 306.8483986360388, 186.4981274825979, 307.30233187175406, 307.03077498069547, 307.61373377363174, 306.79919556224615, 304.3491920856028, 305.9651362692947, 308.9631834880155, 303.90790160420096, 306.08440903322014, 308.8584683706236, 305.4645758508518, 305.5814702061904, 48.891063107776745, 306.54894169039335, 307.04921474914823, 307.786414182452, 308.72792760577215, 304.0112463548104, 305.00945437686073, 302.7404919406677, 306.76512962983736, 306.4038530057744, 305.38919643089315, 307.438971868083, 309.71342187103164, 304.4528808204421, 305.6775377216935, 306.90342587475106, 306.4969054553938, 307.7378294420925, 307.36070190108114, 307.9799805544808, 305.0080155747858, 306.1838535214231, 305.7712302257501, 307.08454818681054, 308.64522925462256, 305.56595393317497, 305.160032553894, 310.3713018255328, 308.6110224098019, 309.59921972089876, 304.74197425687385, 306.29902593956564, 306.53408112472874, 307.8614730911799, 308.4875508521205, 305.572535369342, 307.70645057230564, 307.07539483573345, 307.1784309403132, 307.08086370940345, 305.6910574278807, 307.3895200860931, 306.54144970406156, 306.7042915033888, 307.5793801068879, 307.24308419522464, 305.04278076783953, 307.6369490561535, 305.05387087003254, 308.7821832297253, 305.10640693339946]
301.1521783271539
301.1521783271539
Evaluation over 100 Episodes: 301.15 
MISSION COMPLETED
Score: 301.15 reached at Episode: 164
```
## Evaluation
```buildoutcfg
Score over 100 episodes [305.8996019321888, 308.71932969178835, 306.1439630208252, 308.021447194717, 200.87596506851582, 308.0779480480911, 305.18263351715433, 309.71078606766696, 307.22007435652375, 307.71855012959406, 309.5254770204139, 307.79281255145145, 308.11600955072026, 307.42003484531415, 307.9735337722729, 307.1516310458529, 306.6445770617056, 308.195140676199, 306.8309508523074, 304.8744319317161, 308.64944942032577, 308.4514209847252, 307.2757454775874, 306.6617735800721, 303.3996953481711, 307.07420276580183, 306.5841593034918, 307.5306164924177, 306.2383592143729, 307.0983031856983, 306.32049692798716, 307.866195133915, 306.0279993372719, 305.08054503163686, 306.94001194289206, 305.00717142767945, 305.3108359263642, 307.83466443250313, 307.9564651576301, 306.801077652323, 306.25124603442794, 306.6524945200084, 305.1726861433923, 308.19768867388393, 306.13401908403836, 307.4367627492707, 308.79298681824326, 306.9736248821809, 306.16176525038026, 309.1238268514688, 304.7607486461902, 304.56749884843646, 306.6393663016133, 305.43082544618375, 306.28672682441845, 309.31608847354863, 306.61371584376826, 306.74232180156133, 307.9682448708077, 307.18720961098916, 306.80854555454346, 307.80974846447685, 307.98764737078557, 308.5461637041163, 308.01864327693505, 305.50172449303113, 308.3366967125949, 307.37783732164763, 305.41107568879073, 310.04353202355713, 308.2704934116687, 305.7969960030018, 310.2722596531688, 307.13584186068186, 305.6069627071644, 305.8935206214335, 306.56987344684336, 305.6009266064592, 304.5623741330746, 305.9017072385191, 304.842071780579, 308.640323767354, 302.9353184724574, 308.5820420499812, 306.07032141377124, 307.54611841820144, 307.0185154858187, 308.70374657839807, 307.41020115489005, 306.315764244193, 305.8155368926623, 307.30470537742514, 307.9172614385465, 308.4924032217528, 306.2231954398935, 307.5106815382572, 306.8980168565896, 305.99156820160795, 307.56242880150245, 309.25469771620004]
MEAN: 305.9510142389527
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

To run hyper parameter optimization run the file `tuneSAC.py`  
To run training & test checkout this [colab notebook](https://colab.research.google.com/drive/1gnt05TNOoklUgUQsMRj_EsfxZ5Ww9U2I?usp=sharing)


### Video of trained model
[Watch](https://youtu.be/7PJFJWpD-sM)


### Contributors
- [Mathias Åsberg](https://www.linkedin.com/in/imathias/) 

[![Deep Learning Labs AI ](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/small-banner.jpg)](https://nextgrid.ai/dll)

## Nextgrid AI Seed program

Applications for [Nextgrid](https://nextgrid.ai) AI seed program is now open. We are looking for briliant business & people that are building AI-first solutions. Read more and apply at [AI seed program](https://nextgrid.ai/seed/)

[![Nextgrid Partners](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/partner-banner.jpg)](https://nextgrid.ai/partners/)
