import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from collections import namedtuple
from collections import deque
import numpy as np
import os
import gzip
import pickle

def Q_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        
        
class ReplayBuffer:

    # TODO: implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self,history_length=0):
        
        self.history_length = history_length
        
        self._data = namedtuple(
            "ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"]
        )
        self._data = self._data(
            states=deque(maxlen=history_length), actions=deque(maxlen=history_length), next_states=deque(maxlen=history_length), 
            rewards=deque(maxlen=history_length), dones=deque(maxlen=history_length)
        )
        
        self.count = 0

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)
        
        if self.count < self.history_length:
            self.count += 1

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_dones,
        )



class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256,device='cpu'):
        super(MLP, self).__init__()
        self.device=device
        self.fc1 = nn.Linear(state_dim, hidden_dim,device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,device=device)
        self.fc5 = nn.Linear(hidden_dim, action_dim,device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc5(x)
        return x



class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        gamma=0.99,
        batch_size=256,
        epsilon_min=0.05,
        epsilon=0.4,
        decay = 0.9995,
        lr=1e-4,
        history_length=100_000,
        device='cpu'
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        
        self.device = device
        
        
        # setup networks
        self.Q = Q.to(self.device)
        self.Q_target = Q_target.to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay

        self.loss_function = torch.nn.HuberLoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        
        self.num_actions = num_actions
        self.step = 0
        self.Q_updateStep = 500
        
    def add_transition(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer
        """
        self.replay_buffer.add_transition(state,action,next_state=next_state,reward=reward,done=terminal)
        
    def replay_bufferFilled(self):
        return self.replay_buffer.count == (self.replay_buffer.history_length - 1)

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update:
        #       2.1 compute td targets and loss
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)

        
        self.replay_buffer.add_transition(state,action,next_state=next_state,reward=reward,done=terminal)
        
        if self.step == 0:
            print("training started!")
            
        self.optimizer.zero_grad()
        
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self.replay_buffer.next_batch(self.batch_size)
        state_batch = torch.tensor(state_batch, dtype=torch.float32,device=self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64,device=self.device).unsqueeze(-1)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32,device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32,device=self.device).unsqueeze(1)  # shape [B, 1]
        terminal_batch = torch.tensor(terminal_batch, dtype=torch.float32,device=self.device).unsqueeze(1)  # shape [B, 1]
        
        q_val = self.Q(state_batch)
        q_val=q_val.gather(1,action_batch)

        with torch.no_grad():

            next_state_vals = self.Q_target(next_state_batch).max(dim=1, keepdim=True)[0]
            
        td_target = reward_batch + self.gamma * (1 - terminal_batch) * next_state_vals
        loss = self.loss_function(q_val,td_target)
        loss.backward()
        self.optimizer.step()
        
        if self.step % self.Q_updateStep == 0: 
            Q_update(self.Q_target,self.Q)

        self.step += 1
        torch.cuda.empty_cache()
        
    def updateEpsilon(self):
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay
        
        

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        state = torch.tensor(state,device=self.device).unsqueeze(0)
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            with torch.no_grad():
                action_id = int(torch.argmax(self.Q(state),dim=-1).item())
        else:
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = torch.randint(0,self.num_actions,size=(1,1)).item()

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
 
        