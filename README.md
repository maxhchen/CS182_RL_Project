# CS182_RL_Project

Reinforcement Learning Project for CS W182/282A: Designing, Visualizing and Understanding Deep Neural Networks @ UC Berkeley

## Introduction:
Reinforcement Learning (RL) is a machine learning paradigm centered around training agents to take actions in an environment in order to maximize a reward or goal. One research direction focuses on the ability of an agent to “generalize” what they have learned in one environment to perform well in similar yet novel environments. For instance, an agent playing a game trains to survive a sequence of levels while maximizing its score before being tested on unseen levels. We benchmark the agent’s performance as the score earned by the agent on unseen test levels.

Ideally, an agent should be able to learn not only how to “survive” a level and reach the end without hitting obstacles, but also to optimize their score by eating fruit and avoiding non-fruit objects, and on different, unseen levels as well. This is important not only in the pursuit of more intelligent agents that can handle more tasks, but also to ensure that an agent is truly learning skills and behaviors independent of their environment.

## Modifications:
Our work focuses on entropy regularization and noise regularization techniques. Nascent research has emerged suggesting entropy regularization -- finetuning and scheduling penalties on the entropy of the policy distribution -- can lead to improved convergence rates in natural policy gradient algorithms. We seek to experimentally confirm these results while also exploring its impact on agent generalizability. Additionally, we explore noise regularization, a novel technique (inspired by dropout) introducing random perturbations to the policy distribution in an effort to build redundancy and prevent overfitting to the specific parameters of training environments.

## Training or testing agents
First, change directory to `train_procgen`

If you want to train a new agent:
* Use `nohup python3 -m train --start_level=0 --num_levels=500 --high_entropy=['False', 'True] --scheduler=['none', 'linear', 'piecewise', 'exponential'] --log_dir=NAME`

If you want to test an agent:
* Use `nohup python3 -m train --start_level=500 --num_levels=100 --high_entropy=['False', 'True] --scheduler=['none', 'linear', 'piecewise', 'exponential'] --log_dir=NAME --load_path=FOLDER/NAME/checkpoints/00305`

#### Command Parameters

`high_entropy` = whether to run an agent with an initial `ent_coeff` of 0.01 (low, `False`) or 0.1 (high, `True`)

`scheduler` = whether to run an agent with a linear, piecewise step, or exponential scheduler for `ent_coeff` decay
  * `high_entropy=False` = schedulers will decay from `ent_coeff=1e-2` to `ent_coeff=1e-5`
  * `high_entropy=True` = schedulers will decay from `ent_coeff=1e-1` to `ent_coeff=1e-4`

`log_dir` = file path to save results

`load_path` = file path to existing model

## Contents

- `train_procgen`: Files from train_procgen required to train/run agents
  - `training_runs`: Checkpoints and Progress for Training Runs
  - `test_runs`: Checkpoints and Progress for Test Runs
  - `preliminary_runs`: Initial Experimentation
  - `noise` : Experiments with noise
    - `dirichlet_noise`: Dirichlet Distribution Modification
    - `gaussian_noise`: Gaussian Distribution Modification
- `README.md` : You are here!
- `requirements.txt` : Required Python Packages

## Contributors

- Maxwell Chen [(@maxhchen)][maxwell]
- Abinav Routhu [(@abinavcal)][abinav]
- Jason Lin [(@jasonlin18)][jason]

[maxwell]: https://github.com/maxhchen
[abinav]: https://github.com/abinavcal
[jason]: https://github.com/jasonlin18
