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
MAX_EPISODES = 1000
MAX_STEPS = 2000
SINGLE_WORKER = False

if SINGLE_WORKER:
    environment_platforms = {
        'win32': 'Reacher_Windows_x86_64_single/Reacher.exe',
        'linux': 'Reacher_Linux/Reacher.x86_64'
    }
else:
    environment_platforms = {
        'win32': 'Reacher_Windows_x86_64/Reacher.exe',
        'linux': 'Reacher_Linux/Reacher.x86_64'
    }

np.random.seed(SEED)


def train_agent(max_episodes=MAX_EPISODES, max_t=MAX_STEPS, print_every=100):
    scores = []
    scores_deque = deque(maxlen=print_every)
    running_mean_100 = 0.0
    t = trange(max_episodes, desc='Running mean score: {:.3f}'.format(running_mean_100))

    for idx in t:
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations
        score = np.zeros(len(env_info.agents))

        for _ in range(max_t):
            action = agent.act(state)
            env_info = env.step(action.flatten())[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if any(done):
                break

        scores_deque.append(score)
        scores.append(score.mean())
        running_mean_100 = np.mean(scores_deque)

        t.set_description('Running mean score: {:.3f}'.format(running_mean_100))

        if running_mean_100 >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(idx+1,
                                                                                         np.mean(scores_deque)))
            current_dir = os.path.dirname(os.path.realpath(__file__))
            torch.save(agent.actor_local.state_dict(), os.path.join(current_dir, 'checkpoint_actor.pth'))
            torch.save(agent.critic_local.state_dict(), os.path.join(current_dir, 'checkpoint_critic.pth'))
            break

    env.close()

    return scores


env_dir = os.path.dirname(os.path.realpath(__file__))
relative_path = environment_platforms[sys.platform]
env_path = os.path.join(env_dir, relative_path)
assert os.path.exists(env_path), 'The Unity Reacher environment could not be found'
env = UnityEnvironment(file_name=env_path)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

state_size, action_size = brain.vector_observation_space_size, brain.vector_action_space_size
agent = Agent(state_size=state_size, action_size=action_size, random_seed=SEED)

training_scores = train_agent()

plt.figure()
plt.plot(np.arange(1, len(training_scores) + 1), training_scores)
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
