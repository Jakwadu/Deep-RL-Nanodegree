import os
import sys
from unityagents import UnityEnvironment
import numpy as np
from BananaAgent import Agent
from matplotlib import pyplot as plt
from tqdm import trange
from argparse import ArgumentParser

MAX_EPISODES = 1000
EPSILON = 1.0
EPSILON_DECAY = 0.995
environment_platforms = {
    'win32':'Banana_Windows_x86_64/Banana.exe',
    'linux':'Banana_Linux/Banana.x86_64'
}

np.random.seed(123)

def run_episode(env, epsilon, train=True):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # Configure the environment for training or evaluation
    env_info = env.reset(train_mode=train)[brain_name]
    state = env_info.vector_observations[0]            
    score = 0 
    # Start loop                                         
    while True:
        action = agent.act(state, eps=epsilon)   
        env_info = env.step(np.int32(action))[brain_name]        
        next_state = env_info.vector_observations[0]   
        reward = env_info.rewards[0]                   
        done = env_info.local_done[0]
        # The 'step' call is only required if training
        if train:                 
            agent.step(state, action, reward, next_state, done)
        score += reward                                
        state = next_state                             
        if done:
            break
    return score

def train(max_episodes=MAX_EPISODES, epsilon=EPSILON):
    env = UnityEnvironment(file_name=banana_env_path)

    training_scores = []
    running_mean_scores = []
    running_mean_100 = 0.0
    current_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_path = os.path.join(current_dir, 'checkpoint.pth')

    print('### Starting training loop ...')

    environment_solved = False
    t = trange(max_episodes, desc='Running mean score: {:.3f}'.format(running_mean_100))
    pass_criterion_counter = 0
    
    for _ in t:
        score = run_episode(env, epsilon)
        training_scores.append(score)
        if len(training_scores) < 100:
            running_mean_100 = np.mean(training_scores)
        else:
            running_mean_100 = np.mean(training_scores[-100:])
        running_mean_scores.append(running_mean_100)
        t.set_description('Running mean score: {:.3f} | Epsilon: {:.3f}'.format(running_mean_100, epsilon))

        if running_mean_100 >= 13.0:
            pass_criterion_counter += 1
            # Check that running meas is greater than 13 for 100 consecutive episodes
            if pass_criterion_counter >= 100:
                environment_solved = True
                break
        else:
            pass_criterion_counter = 0
        
        if epsilon > 0.01:
            epsilon *= EPSILON_DECAY

    if environment_solved:
        print(f'### Your Banana Agent is ready for action! Score: {running_mean_100}')

    agent.save_model_weights(checkpoint_path)
    print(f'### Model saved to {checkpoint_path}')

    env.close()

    plt.title('Scores')
    plt.plot(training_scores)
    plt.plot(running_mean_scores)
    plt.legend(['Episodic', 'Running Mean'])
    plt.show()

def evaluate():
    env = UnityEnvironment(file_name=banana_env_path)

    print('### Starting evaluation loop ...')

    scores = []
    score = 0
    t = trange(100, desc='Latest score: {:.3f}'.format(score))
    
    for episode in t:
        score = run_episode(env, 0.0)
        scores.append(score)
        t.set_description('Latest score: {:.3f}'.format(score))
    print('### Mean score over 100 episodes: {:3f}'.format(np.mean(scores)))    


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(description='Exercise a Q-Learning in the Unity Banana Environment')
    parser.add_argument('--evaluate', help='Test a pre-trained agent')
    parser.add_argument('--environment_path', help='Path to the Unity Banana Environment', type=str, dest='banana_env')
    parser.add_argument('--checkpoint', help='Path to pre-trained weights', type=str, dest='checkpoint')
    args = parser.parse_args()
    
    # Find th Banana environment
    if args.banana_env:
        banana_env_path = args.banana_env
    else:
        print('### Using default environment path')
        current_dir = os.path.dirname(os.path.realpath(__file__))
        relative_path = environment_platforms[sys.platform]
        banana_env_path = os.path.join(current_dir, relative_path)
    assert os.path.exists(banana_env_path), 'The Unity Banana environment could not be found'
    
    # Create Q learning agent
    agent = Agent(37, 4, 123)
    checkpoint = args.checkpoint
    if checkpoint:
        agent.load_model_weights(checkpoint)
    
    # Interact with the environment
    if args.evaluate:
        assert checkpoint, 'No model weights provided'
        agent.evaluation_mode()
        evaluate()
    else:
        train()
