[![Nextgrid Artificial Intelligence](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/big-banner.jpg)](https://nextgrid.ai)



# MountainCarContinuous-v0

### Solved at Episode 9
![MountainCarContinues SEC](MountainCarContinuous.gif)
Mission: Take top position in all OpenAI leaderboard. By [Nextgrid](https://nextgrid.ai) and [Deep Learning Labs](https://nextgrid.ai/deep-learning-labs/) find all attempts [here](https://github.com/nextgrid/deep-learning-labs-openAI). The MountainCarContinuous-v0 was **solved on 9 Episodes** 

## Solution
Optuna & SAC: Soft Actor Critic (SAC) Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.

## HyperParameter tuning
Best HyperParameters was identified with help of Optuna
See [tuneSAC.py](tuneSAC.py) for complete HP code.  

**Stack**:  
`Stables-baselines3`   
`Optuna`  
`Mysql`  
`Docker`  
`Kubernetes` 

### Sampler
Optuna sampler configuration
```buildoutcfg
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
```
### Trials
Did run +400 trials where trial 56 delivered the best result solving the BipedalWalker problem on 70 episodes. After manually tweaking some functions and HyperParameters it came down to 9 Episodes. Colab notebook: [MountainCarContinuous-v0_SAC_Tensorboard.ipynb](https://colab.research.google.com/drive/1cVTbyz5bZ8-d573pV6jzbLqL1_JyA6Os?usp=sharing)  

```
FrozenTrial(number=56, value=70.0, datetime_start=datetime.datetime(2020, 12, 15, 7, 50, 45), datetime_complete=datetime.datetime(2020, 12, 15, 8, 40, 35), params={'batch_size': 512, 'buffer_size': 10000, 'gamma': 0.95, 'learning_starts': 10, 'log_std_init': -1.4627, 'lr': 0.000969826, 'net_arch': 'big', 'sde_sample_freq': 32, 'tau': 0.02, 'train_freq': 64}```
```
### HyperParameters 

hp = {'batch_size': 256, 'buffer_size': 10000, 
      'gamma': 0.95, 'learning_starts': 10, 'log_std_init': -1.4627, 
      'lr': 0.000969826, 'net_arch': [256, 256], 
      'sde_sample_freq': 32, 'tau': 0.02, 'train_freq': 64, 
      'ent_coef': 'auto', 'target_entropy': 'auto', 'gradient_steps': 64, 'use_sde': True}


model = SAC(
    MlpPolicy,
    env,
    gamma=hp["gamma"],
    learning_rate=hp["lr"],
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
    policy_kwargs=dict(log_std_init=hp["log_std_init"], net_arch=hp["net_arch"]),
    tensorboard_log=logs_base_dir,
    verbose=0
)
```

## Training
```
Reward threshold achieved
Evaluating model....10 EP
Score over 100 episodes [-3.4188024079515817, -3.4326296741131266, -3.42555676285335, -3.435651723482596, -3.4332032374709027, -3.3988742106748964, -3.418274480685373, -3.4250138767376335, -3.413258623679077, -3.434310815791979]
-3.4235575813440517
-3.4235575813440517
Evaluation over 10 Episodes: -3.42 
Episodes: 5
Num steps: 1536
Mean Episode reward: 94.00 
Last Episode reward: 94.37 
=========== NEXTGRID.AI ================
Reward threshold achieved
Evaluating model....10 EP
Score over 100 episodes [-3.0924918327082334, -3.101544887619765, -3.1315010235412117, -3.0989547697928392, -3.1183284348584266, -3.101808306399921, -3.1189775776984816, -3.094490157502583, -3.113263910403179, -3.1018157584030264]
-3.107317665892767
-3.107317665892767
Evaluation over 10 Episodes: -3.11 
Episodes: 5
Num steps: 1792
Mean Episode reward: 94.00 
Last Episode reward: 94.37 
=========== NEXTGRID.AI ================
Reward threshold achieved
Evaluating model....10 EP
Score over 100 episodes [-6.109398655496566, -6.1305606116603375, -6.1245172953475695, -6.262865590130196, -6.116762305285978, -6.115488679238101, -6.180696179724453, -6.129904378446748, -6.122790866582586, -6.148751341783986]
-6.144173590369652
-6.144173590369652
Evaluation over 10 Episodes: -6.14 
Episodes: 6
Num steps: 2048
Mean Episode reward: 90.00 
Last Episode reward: 89.73 
=========== NEXTGRID.AI ================
Reward threshold achieved
Evaluating model....10 EP
Score over 100 episodes [-9.125112478846733, -9.694569151777277, -9.311675293346307, -9.116650645521181, -9.916658691976211, -9.46268133899274, -9.126335821917396, -9.528454342612946, -9.086573781464164, -9.136242229309639]
-9.35049537757646
-9.35049537757646
Evaluation over 10 Episodes: -9.35 
Episodes: 6
Num steps: 2304
Mean Episode reward: 90.00 
Last Episode reward: 89.73 
=========== NEXTGRID.AI ================
Reward threshold achieved
Evaluating model....10 EP
Score over 100 episodes [-2.0545827442494806, -1.6744464216613295, -5.237191359259563, -4.4278380721768364, -4.00949791882569, -1.6635038375789999, -3.2876127377323363, -4.75562236489676, -1.9337454641112688, -6.002482690644683]
-3.5046523611136946
-3.5046523611136946
Evaluation over 10 Episodes: -3.50 
Episodes: 7
Num steps: 2560
Mean Episode reward: 83.00 
Last Episode reward: 83.22 
=========== NEXTGRID.AI ================
Episodes: 7
Num steps: 2816
Mean Episode reward: 83.00 
Last Episode reward: 83.22 
=========== NEXTGRID.AI ================
Episodes: 7
Num steps: 3072
Mean Episode reward: 83.00 
Last Episode reward: 83.22 
=========== NEXTGRID.AI ================
Episodes: 9
Num steps: 3328
Mean Episode reward: 95.00 
Last Episode reward: 94.96 
=========== NEXTGRID.AI ================
Reward threshold achieved
Evaluating model....10 EP
Score over 100 episodes [90.41553479743453, 91.27321934460024, 94.52903978131712, 90.97865399231645, 94.52852509917625, 87.01370128748113, 86.87148304490995, 91.20285198986431, 94.52080330016774, 92.82585978649306]
91.41596724237608
91.41596724237608
Evaluation over 10 Episodes: 91.42 
Evaluating model....100 EP
Score over 100 episodes [-22.406309617012454, 91.23118720634554, 90.80338655697327, 87.83383576750303, 92.3129050511798, 90.50729760672881, 92.70547383448584, 94.4899284096797, 94.57047233965685, 91.19639734633586, 94.60127186931847, 91.81095847490529, 92.94737593628327, 92.41964131731203, 94.31918702079737, 92.93644214948645, 91.26488462696278, 92.49914047828263, 87.39405891801756, 94.55265073257407, 91.29318826547222, 86.4716583571414, 94.55192487112495, 92.89977726458075, 92.60130995214342, 88.50003472809654, 89.2467786050774, 94.53230440069463, 92.76291770625285, 94.56859515271204, 94.53904909364434, 94.57539221286143, 89.51255874228792, 86.98454427420904, 94.57534435222179, 92.93392393952251, 94.45139700150196, 92.70576545048799, 92.82297008003262, 94.48449246883146, 92.93305363151372, 92.94231472894737, 90.50767148906463, 91.28306361330317, 91.23871671486239, 87.77173030346539, 92.85308771787194, 87.47064831025753, 94.55119191872876, 94.58745442330408, 83.93806331591077, 91.29693635174733, 87.2749520940028, 94.52222356409591, 92.92371988551758, 89.31891766468613, 90.4760293494665, 94.5953170518185, 94.44761738977515, 92.96288200730301, 92.8575983318341, 92.94251979342837, 94.60445472915995, 92.52610018815443, 94.45865504614154, 84.88591257378638, 85.92526342632118, 91.05628400439964, 80.31418781447891, 91.13991536379199, 92.79425913103934, 94.52155056827097, 94.29317785817766, 92.73639300033115, 87.6923769374313, 92.82010918074477, 92.96000868123299, 94.57592895293713, 94.45139350944156, 92.67791988012232, 79.16248820348761, 94.5923636390295, 86.79327114820309, 92.7917125953086, 81.4823173007053, 85.75695886211365, 89.24678054648294, 88.51789715658924, 87.02043225598115, 92.81902699597734, 92.41952015096336, 94.14469953175345, 92.79238135661008, 85.19869893013211, 92.93256938223149, 91.02958852859771, 91.29811541714929, 94.57598001818951, 92.7051813881486, 88.21733635296779]
90.24631033202202
Evaluation over 100 Episodes: 90.25 
MISSION COMPLETED
```
## Evaluation
```buildoutcfg
Score over 100 episodes [84.37790690216903, 89.4800125136396, 92.7901638345887, 94.45839180941672, 94.6037733748626, 90.95427780550925, 80.38014182467015, 94.57551094382035, 92.9475503134798, 94.42496526817203, 91.22683068048137, 94.52765685099092, 94.48239839026485, 94.59363335882554, 94.53758894507048, 92.70875992142568, 91.28303075529105, 91.27708154712577, 92.7909123553711, 92.93713135385903, 94.57616813523997, 91.88099406247109, 94.4759931368866, 92.8511292634302, 92.935621263535, 94.60695973838382, 92.85864888068218, 92.92228180976316, 88.57417691502955, 92.28672369168206, 91.54960750440446, 91.28941098680637, 94.4911219647742, 85.28904952741885, 94.56592238646176, 94.59133069053021, 94.44752691935741, 94.59849734240599, 94.52154103913031, 94.58171254669419, 94.04741134224642, 94.57531450549688, 92.41955527464208, 92.9475351971371, 89.57536098175908, 94.54013395607807, 94.5892384631154, 94.58835098225025, 91.29058752580842, 92.94751794532075, 92.79154937357615, 88.22911588681728, 92.52541703583513, 91.19426108180147, 89.2814281461982, 87.5970066883854, 92.7374026945359, 92.93959922363672, 92.89084966145086, 92.60142938846337, 92.50113229043336, 92.73352924845744, 91.20201016356384, 91.19574136783035, 94.54058009228075, 89.39682732729558, 94.54043359614016, 89.44915845396316, 92.92893887820885, 89.5115751611008, 94.57798904788076, 91.16646001127162, 94.56861177545856, 90.38682129877394, 94.58174190341059, 92.73632473358316, 92.95766649700543, 94.59527751216746, 92.9539595874474, 92.70730955549473, 91.85764033900402, 94.57529262093561, 86.81925419564263, 92.89323454239428, 92.88741099770209, 94.4875248468398, 92.52567505079291, 92.93870874156885, 92.87268553678877, 91.0832724088593, 92.65128873448286, 90.978463983295, 89.57905458083532, 94.55305813531527, 94.57872866967641, 94.591820384862, 92.94720301371711, 87.31553267359043, 92.89029764816596, 94.58395816611574]
MEAN: 92.38403395681027
```

### How to run

To run hyper parameter optimization run the file `tuneSAC.py`  
To run training & test checkout this [colab notebook](https://colab.research.google.com/drive/1cVTbyz5bZ8-d573pV6jzbLqL1_JyA6Os?usp=sharing)


### Video of trained model
[Watch](https://youtu.be/_mEHElqOmbM)


### Contributors
- [Mathias Åsberg](https://www.linkedin.com/in/imathias/) 

[![Deep Learning Labs AI ](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/small-banner.jpg)](https://nextgrid.ai/dll)

## Nextgrid AI Seed program

Applications for [Nextgrid](https://nextgrid.ai) AI seed program is now open. We are looking for briliant business & people that are building AI-first solutions. Read more and apply at [AI seed program](https://nextgrid.ai/seed/)

[![Nextgrid Partners](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/partner-banner.jpg)](https://nextgrid.ai/partners/)
