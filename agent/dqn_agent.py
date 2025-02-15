import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections
import random
import dill as pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MemoryBuffer(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.trans_counter = 0
        self.buffer = collections.deque(maxlen=self.memory_size)
        self.transition = collections.namedtuple("Transition",
                                                 field_names=["state", "action", "reward", "new_state", "terminal"])

    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size
        transitions = random.sample(self.buffer, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float().to(device)
        terminals = torch.from_numpy(
            np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, new_states, terminals


class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


class Agent(object):
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996, epsilon_end=0.01, mem_size=1000000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_dec = epsilon_dec  # Epsilon decay rate
        self.epsilon_min = epsilon_end  # Minimum epsilon
        self.batch_size = batch_size
        self.memory = MemoryBuffer(mem_size)

    def save(self, state, action, reward, new_state, done):
        self.memory.save(state, action, reward, new_state, done)

    def choose_action(self, state):
        rand = np.random.random()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_func.eval()
        with torch.no_grad():
            action_values = self.q_func(state)
        self.q_func.train()
        if rand > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice([i for i in range(self.action_size)])

    def reduce_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self):
        raise Exception("Not implemented")

    def save_model(self, path):
        torch.save(self.q_func.state_dict(), path)

    def load_saved_model(self, path):
        self.q_func = QNN(self.state_size, self.action_size, 42).to(device)
        self.q_func.load_state_dict(torch.load(path))
        self.q_func.eval()


class DoubleQAgent(Agent):
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996, epsilon_end=0.01, mem_size=1000000, replace_q_target=100):
        super().__init__(state_size, action_size, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
                         lr=lr, epsilon_dec=epsilon_dec, epsilon_end=epsilon_end, mem_size=mem_size)
        self.replace_q_target = replace_q_target
        self.q_func = QNN(state_size, action_size, 42).to(device)
        self.q_func_target = QNN(state_size, action_size, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)

    def learn(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states).gather(1, actions)
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.memory.trans_counter % self.replace_q_target == 0:
            self.q_func_target.load_state_dict(self.q_func.state_dict())
        self.reduce_epsilon()

    def save_model(self, path):
        super().save_model(path)
        torch.save(self.q_func_target.state_dict(), path + '.target')

    def load_saved_model(self, path):
        super().load_saved_model(path)
        self.q_func_target = QNN(self.state_size, self.action_size, 42).to(device)
        self.q_func_target.load_state_dict(torch.load(path + '.target'))
        self.q_func_target.eval()
