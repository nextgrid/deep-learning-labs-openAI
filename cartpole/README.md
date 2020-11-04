[![Nextgrid Artificial Intelligence](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/big-banner.jpg)](https://nextgrid.ai)

# CartPole

Mission: Take top position in all OpenAI leaderboards. By [Deep Learning Labs](https://nextgrid.ai/deep-learning-labs/) find all attempts [here](https://github.com/nextgrid/deep-learning-labs-openAI)

## Solution

We used Pytorch and DQN with Actor critic.  
Optima was used to find the best hyper parameters

```
gamma = 0.9511969141631759
learning_rate = 0.017200527912726204
np_float = "float32"
seed = 996
nn_size = 64
```

### Stack

```
Pytorch
Gym
Optima
```

### How to run

To run hyper parameter optimization run the file `optima.py`  
To run training run `cartpole.py` to enable rendering add `--render` like `/path/cartpole.py --render`

### Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/a0oA5VmVFhQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### GIF Trained model

![CartPole Recording](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/gif/cart-pole.gif)

### Plotting

![CartPole Plotting](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/img/Screenshot%202020-11-04%20at%2015.44.55.png)

### Output from training

```
Episode 1	Last reward: 8.00	Average reward: 9.90
Episode 2	Last reward: 29.00	Average reward: 10.86
Episode 3	Last reward: 24.00	Average reward: 11.51
Episode 4	Last reward: 14.00	Average reward: 11.64
Episode 5	Last reward: 17.00	Average reward: 11.90
Episode 6	Last reward: 76.00	Average reward: 15.11
Episode 7	Last reward: 78.00	Average reward: 18.25
Episode 8	Last reward: 27.00	Average reward: 18.69
Episode 9	Last reward: 11.00	Average reward: 18.31
Episode 10	Last reward: 26.00	Average reward: 18.69
Episode 11	Last reward: 40.00	Average reward: 19.76
Episode 12	Last reward: 33.00	Average reward: 20.42
Episode 13	Last reward: 41.00	Average reward: 21.45
Episode 14	Last reward: 54.00	Average reward: 23.08
Episode 15	Last reward: 32.00	Average reward: 23.52
Episode 16	Last reward: 43.00	Average reward: 24.50
Episode 17	Last reward: 72.00	Average reward: 26.87
Episode 18	Last reward: 38.00	Average reward: 27.43
Episode 19	Last reward: 26.00	Average reward: 27.36
Episode 20	Last reward: 93.00	Average reward: 30.64
Episode 21	Last reward: 91.00	Average reward: 33.66
Episode 22	Last reward: 89.00	Average reward: 36.42
Episode 23	Last reward: 33.00	Average reward: 36.25
Episode 24	Last reward: 63.00	Average reward: 37.59
Episode 25	Last reward: 232.00	Average reward: 47.31
Episode 26	Last reward: 38.00	Average reward: 46.84
Episode 27	Last reward: 39.00	Average reward: 46.45
Episode 28	Last reward: 47.00	Average reward: 46.48
Episode 29	Last reward: 23.00	Average reward: 45.31
Episode 30	Last reward: 30.00	Average reward: 44.54
Episode 31	Last reward: 28.00	Average reward: 43.71
Episode 32	Last reward: 42.00	Average reward: 43.63
Episode 33	Last reward: 56.00	Average reward: 44.25
Episode 34	Last reward: 46.00	Average reward: 44.33
Episode 35	Last reward: 134.00	Average reward: 48.82
Episode 36	Last reward: 105.00	Average reward: 51.63
Episode 37	Last reward: 39.00	Average reward: 51.00
Episode 38	Last reward: 135.00	Average reward: 55.20
Episode 39	Last reward: 84.00	Average reward: 56.64
Episode 40	Last reward: 28.00	Average reward: 55.20
Episode 41	Last reward: 160.00	Average reward: 60.44
Episode 42	Last reward: 104.00	Average reward: 62.62
Episode 43	Last reward: 92.00	Average reward: 64.09
Episode 44	Last reward: 259.00	Average reward: 73.84
Episode 45	Last reward: 127.00	Average reward: 76.49
Episode 46	Last reward: 45.00	Average reward: 74.92
Episode 47	Last reward: 31.00	Average reward: 72.72
Episode 48	Last reward: 54.00	Average reward: 71.79
Episode 49	Last reward: 136.00	Average reward: 75.00
Episode 50	Last reward: 96.00	Average reward: 76.05
Episode 51	Last reward: 72.00	Average reward: 75.85
Episode 52	Last reward: 147.00	Average reward: 79.40
Episode 53	Last reward: 83.00	Average reward: 79.58
Episode 54	Last reward: 78.00	Average reward: 79.50
Episode 55	Last reward: 62.00	Average reward: 78.63
Episode 56	Last reward: 90.00	Average reward: 79.20
Episode 57	Last reward: 249.00	Average reward: 87.69
Episode 58	Last reward: 108.00	Average reward: 88.70
Episode 59	Last reward: 101.00	Average reward: 89.32
Episode 60	Last reward: 91.00	Average reward: 89.40
Episode 61	Last reward: 163.00	Average reward: 93.08
Episode 62	Last reward: 67.00	Average reward: 91.78
Episode 63	Last reward: 153.00	Average reward: 94.84
Episode 64	Last reward: 78.00	Average reward: 94.00
Episode 65	Last reward: 290.00	Average reward: 103.80
Episode 66	Last reward: 155.00	Average reward: 106.36
Episode 67	Last reward: 292.00	Average reward: 115.64
Episode 68	Last reward: 372.00	Average reward: 128.46
Episode 69	Last reward: 500.00	Average reward: 147.03
Episode 70	Last reward: 500.00	Average reward: 164.68
Episode 71	Last reward: 487.00	Average reward: 180.80
Episode 72	Last reward: 248.00	Average reward: 184.16
Episode 73	Last reward: 223.00	Average reward: 186.10
Episode 74	Last reward: 312.00	Average reward: 192.40
Episode 75	Last reward: 500.00	Average reward: 207.78
Episode 76	Last reward: 500.00	Average reward: 222.39
Episode 77	Last reward: 500.00	Average reward: 236.27
Episode 78	Last reward: 500.00	Average reward: 249.45
Episode 79	Last reward: 500.00	Average reward: 261.98
Episode 80	Last reward: 500.00	Average reward: 273.88
Episode 81	Last reward: 500.00	Average reward: 285.19
Episode 82	Last reward: 500.00	Average reward: 295.93
Episode 83	Last reward: 500.00	Average reward: 306.13
Episode 84	Last reward: 500.00	Average reward: 315.83
Episode 85	Last reward: 500.00	Average reward: 325.03
Episode 86	Last reward: 500.00	Average reward: 333.78
Episode 87	Last reward: 500.00	Average reward: 342.09
Episode 88	Last reward: 500.00	Average reward: 349.99
Episode 89	Last reward: 500.00	Average reward: 357.49
Episode 90	Last reward: 500.00	Average reward: 364.62
Episode 91	Last reward: 500.00	Average reward: 371.38
Episode 92	Last reward: 123.00	Average reward: 358.97
Episode 93	Last reward: 498.00	Average reward: 365.92
Episode 94	Last reward: 500.00	Average reward: 372.62
Episode 95	Last reward: 110.00	Average reward: 359.49
Episode 96	Last reward: 500.00	Average reward: 366.52
Episode 97	Last reward: 500.00	Average reward: 373.19
Episode 98	Last reward: 500.00	Average reward: 379.53
Episode 99	Last reward: 500.00	Average reward: 385.55
Episode 100	Last reward: 500.00	Average reward: 391.28
Solved at Episode 0 With avg score of 204.2 over the following 100 Episodes
204.2
[8.0, 29.0, 24.0, 14.0, 17.0, 76.0, 78.0, 27.0, 11.0, 26.0, 40.0, 33.0, 41.0, 54.0, 32.0, 43.0, 72.0, 38.0, 26.0, 93.0, 91.0, 89.0, 33.0, 63.0, 232.0, 38.0, 39.0, 47.0, 23.0, 30.0, 28.0, 42.0, 56.0, 46.0, 134.0, 105.0, 39.0, 135.0, 84.0, 28.0, 160.0, 104.0, 92.0, 259.0, 127.0, 45.0, 31.0, 54.0, 136.0, 96.0, 72.0, 147.0, 83.0, 78.0, 62.0, 90.0, 249.0, 108.0, 101.0, 91.0, 163.0, 67.0, 153.0, 78.0, 290.0, 155.0, 292.0, 372.0, 500.0, 500.0, 487.0, 248.0, 223.0, 312.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 123.0, 498.0, 500.0, 110.0, 500.0, 500.0, 500.0, 500.0, 500.0]
```

![dasd](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/small-banner.jpg)

## Nextgrid AI Seed program

Applications for [Nextgrid](https://nextgrid.ai) AI seed program is now open. We are looking for briliant business & people that are building AI-first solutions. Read more and apply at [AI seed program](https://nextgrid.ai/seed/)

![dasd](https://storage.googleapis.com/nextgrid_github_repo_visuals/Github%20Graphics%20/partner-banner.jpg)
