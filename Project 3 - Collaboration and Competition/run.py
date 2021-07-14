import os
import sys
import torch
from tqdm import trange
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from ddpg_agent import Agent

SEED = 123
PRE_TRAIN_EPISODES = 0
MAX_EPISODES = 2000
MAX_STEPS = 2000
USE_MADDPG = False

environment_platforms = {
    'win32': 'Tennis_Windows_x86_64/Tennis.exe',
    'linux': 'Tennis_Linux/Tennis.x86_64'
}

np.random.seed(SEED)


def train_agent(max_episodes=MAX_EPISODES, max_t=MAX_STEPS):
    scores = []
    running_mean_scores = []
    scores_deque = deque(maxlen=100)
    running_mean_100 = 0.0

    text = 'Running mean score: {:.3f}'.format(running_mean_100)
    t = trange(max_episodes, desc=text)

    for idx in t:
        agent.reset()
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = np.zeros(len(env_info.agents))
        learning_enabled = (idx >= PRE_TRAIN_EPISODES)

        for _ in range(max_t):
            if learning_enabled:
                action = agent.act(state)
                env_action = np.concatenate([action[0], action[1]])
            else:
                action = np.random.randn(2, action_size)
                env_action = np.clip(action, -1, 1)
            env_info = env.step(env_action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, reward, next_state, done, learning_enabled=learning_enabled)
            score += reward
            state = next_state
            if any(done):
                break

        max_score = score.max()
        scores_deque.append(max_score)
        scores.append(max_score)
        running_mean_100 = np.mean(scores_deque)
        running_mean_scores.append(running_mean_100)

        text = 'Running mean score: {:.3f}'.format(running_mean_100)
        t.set_description(text)

        if running_mean_100 >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(idx+1,
                                                                                         np.mean(scores_deque)))
            current_dir = os.path.dirname(os.path.realpath(__file__))
            torch.save(agent.actor_local.state_dict(), os.path.join(current_dir, 'checkpoint_actor.pth'))
            torch.save(agent.critic_local.state_dict(), os.path.join(current_dir, 'checkpoint_critic.pth'))
            break

    env.close()

    return scores, running_mean_scores


env_dir = os.path.dirname(os.path.realpath(__file__))
relative_path = environment_platforms[sys.platform]
env_path = os.path.join(env_dir, relative_path)
assert os.path.exists(env_path), 'The Tennis environment could not be found'
env = UnityEnvironment(file_name=env_path)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

state_size, action_size = brain.vector_observation_space_size*brain.num_stacked_vector_observations,\
                          brain.vector_action_space_size

agent = Agent(state_size, action_size, random_seed=SEED)

training_scores, running_means = train_agent()

plt.figure()
plt.plot(np.arange(1, len(training_scores) + 1), training_scores)
plt.plot(np.arange(1, len(running_means) + 1), running_means)
plt.legend(['Episodic', 'Running Mean'])
plt.ylabel('Score')
plt.xlabel('Episode')

plt.show()
