# Project 2: Continuous Control

This repository contains a solution to the task of training an agent for the 20 worker version of the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment as specified by the [Continuous Control](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control) project specification. As per the project description, the 20 agent environment is considered solved *if the average score of the twenty agents over the last 100 episodes has a mean of 30+*. The agent used in the training script is based on the DDPG algorithm. The reacher environment has a state size of 33 and action space size of 4, which are reflected in the architctures of the the actor and critic networks. A PPO agent was also implemented but is yet to be completed.

The training script can be executed using the following command:

```
python run.py
```

Python installation prerequisites can be installed using pip:

```
python -m pip install -r requirements.txt
```

*Note: run.py assumes that the Reacher environment root directory is in the same place as the training script (eg. './Reacher_Windows_x86_64/Reacher.exe')*


