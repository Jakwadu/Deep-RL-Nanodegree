# Project 1: Navigation

This is a solution to Project 1 of the Udacity Deep Reinfocement Learning Nanodegree. The project specification can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation). `BananaAgent.py` and `model.py` are based on Deep Q Network excercise found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn). The main script is run from `run.py` and can be executed with the following options:

`--evaluate` - Run the model with pretrained weights.

`--environment_path` - Specify the path to the Banana environment simulator.

`--checkpoint` - Specify the path to pretrained weights.

To run the script in training mode use:

```
python run.py --environment_path /path/to/banana/environment
```

To run the script in evaluation mode with pretrained weights use:

```
python run.py --evaluate --environment_path /path/to/banana/environment --checkpoint /path/to/checkpoint.pth
```

`run.py` can be executed without specifying `--environment_path`, however the following assumptions will be made on the location of the simulator:

**Windows:** `./Banana_Windows_x86_64/Banana.exe`

**Linux:** `./Banana_Linux/Banana.x86_64`

*NOTE: `run.py` will not run in evaluation mode without specifying a checkpoint to be loaded*
