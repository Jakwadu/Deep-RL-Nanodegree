import os
import numpy as np

from model import ActorCriticNetwork

import torch
import torch.optim as optim

MAX_BUFFER_SIZE = 2048
BATCH_SIZE = 128
GAMMA = 0.99  # Reward discount factor
LAMBDA = 0.8  # GAE parameter
EPSILON = 0.2  # Clipping offset
BETA = 0.0  # Parameter for entropy based regularisation
LEARNING_RATE = 1e-4
TRAINING_EPOCHS = 10
REDUCE_LR = False
USE_GAE = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        seed,
        lr=LEARNING_RATE,
        layer_nodes=None,
        n_agents=20
    ):
        # torch.autograd.set_detect_anomaly(True)
        self.n_agents = n_agents

        self.policy = ActorCriticNetwork(state_size, action_size, seed, layer_nodes=layer_nodes).to(device)
        self.optimiser = optim.Adam(self.policy.parameters(), lr=lr)
        self.lr_scheduler = None
        if REDUCE_LR:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimiser, 50, 0.5)
        
        self.states = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.losses = []

    def act(self, state):
        # Get action distribution parameters and sample actions
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        distribution, _ = self.policy(state)

        actions = distribution.sample().view(self.n_agents, -1)
        log_probs = distribution.log_prob(actions).view(self.n_agents, -1)
        log_probs = torch.sum(log_probs, dim=1, keepdim=True)
        
        actions = actions.cpu().numpy()
        log_probs = log_probs.cpu().detach().numpy()
        
        return actions, log_probs

    def step(self, state, log_probs, action, reward, next_state, done):
        self.states.append(state)
        self.log_probs.append(log_probs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        buffer_size = len(self.states)
        if buffer_size >= MAX_BUFFER_SIZE:
            self.learn()

    def learn(self):
        batch_indices = np.arange(0, len(self.states) - 1, BATCH_SIZE)
        np.random.shuffle(batch_indices)

        for _ in range(TRAINING_EPOCHS):
        
            for idx in batch_indices:
                # Create training batches

                all_states = torch.from_numpy(np.vstack(self.states[idx:idx+BATCH_SIZE])).float().to(device)
                all_actions = torch.from_numpy(np.vstack(self.actions[idx:idx+BATCH_SIZE])).long().to(device)
                all_log_probs = torch.from_numpy(np.vstack(self.log_probs[idx:idx+BATCH_SIZE])).float().to(device)
                all_rewards = torch.from_numpy(np.concatenate(self.rewards[idx:idx+BATCH_SIZE])).float().to(device)
                all_next_states = torch.from_numpy(np.vstack(self.next_states[idx:idx + BATCH_SIZE])).float().to(device)
                all_dones = torch.from_numpy(np.concatenate(self.dones[idx:idx + BATCH_SIZE])).float().to(device)

                losses = []

                for offset in range(self.n_agents):
                    indices = list(range(offset, BATCH_SIZE*self.n_agents, self.n_agents))[:len(all_states)]
                    states = all_states[indices]
                    actions = all_actions[indices]
                    log_probs = all_log_probs[indices]
                    rewards = all_rewards[indices]
                    next_states = all_next_states[indices]
                    dones = all_dones[indices]

                    distributions, values = self.policy(states)
                    next_values = self.policy(next_states)[1]

                    # Calculate returns and advantages for training the model
                    returns = torch.zeros_like(values)

                    if USE_GAE:
                        gae = torch.zeros_like(rewards)
                        last_gae = 0
                        for idx_1 in reversed(range(len(rewards))):
                            returns[idx_1] = rewards[idx_1] + GAMMA * (1 - dones[idx_1]) * next_values[idx_1]
                            td_error = rewards[idx_1] + GAMMA * (1 - dones[idx_1]) * next_values[idx_1] - values[idx_1]
                            gae[idx_1] = last_gae * LAMBDA * GAMMA * (1 - dones)[idx_1] + td_error
                            returns[idx_1] = returns[idx_1] + gae[idx_1]
                            last_gae = gae[idx_1]
                        advantages = returns - values
                    else:
                        advantages = torch.zeros_like(rewards)
                        for idx_1 in reversed(range(len(rewards))):
                            returns[idx_1] = rewards[idx_1] + GAMMA * (1 - dones[idx_1]) * next_values[idx_1]
                            td_error = rewards[idx_1] + GAMMA * (1 - dones[idx_1]) * next_values[idx_1] - values[idx_1]
                            advantages[idx_1] = td_error

                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

                    new_probs = distributions.log_prob(actions).view(len(states), -1)
                    new_probs = torch.sum(new_probs, dim=1, keepdim=True).exp()

                    old_probs = log_probs.exp()

                    # Calculate actor loss using PPO with entropy regularisation
                    prob_ratio = new_probs / old_probs
                    raw_loss = prob_ratio * advantages
                    clipped_loss = torch.clamp(prob_ratio, 1.0 - EPSILON, 1.0 + EPSILON) * advantages
                    clipped_surrogate_loss = torch.min(raw_loss, clipped_loss)
                    entropy = -(new_probs * torch.log(old_probs + 1.e-10) +
                                (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))
                    actor_loss = -(clipped_surrogate_loss + BETA * entropy).mean()

                    critic_loss = 0.5 * torch.mean(torch.square(returns - values))

                    agent_loss = actor_loss + critic_loss

                    losses.append(agent_loss)

                mean_loss = sum(losses) / self.n_agents

                self.losses.append(mean_loss.cpu().detach())

                self.optimiser.zero_grad()
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimiser.step()

        # Reset memory buffers for next training episode
        self.states = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def save_model_weights(self, directory):
        policy_path = os.path.join(directory, 'ppo_actor_critic.pth')
        torch.save(self.policy.state_dict(), policy_path)

    def load_model_weights(self, directory):
        policy_path = os.path.join(directory, 'ppo_actor_critic.pth')
        self.policy.load_state_dict(torch.load(policy_path))

    def evaluation_mode(self):
        self.policy.eval()
