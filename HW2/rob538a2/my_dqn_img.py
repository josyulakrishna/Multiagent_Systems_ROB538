import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from tensorboardX import SummaryWriter
from tqdm import tqdm
from my_grid import *

# https://github.com/gxywy/pytorch-dqn/blob/master/dqn.py

class ExperienceReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            # self.buffer.pop(0)
            # self.buffer.append((state, action, reward, next_state, done))
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Net(nn.Module):
    def __init__(self, output):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(6 * 6 * 64 * 16, 512)
        self.fc5 = nn.Linear(512, output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.eval_net = Net(n_actions)
        self.target_net = Net(n_actions)
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=1e-4, alpha=0.95, eps=0.01) ## 10.24 fix alpha and eps
        self.loss_func = torch.nn.MSELoss()
        self.replay_memory = ExperienceReplayBuffer(100000)
        self.steps = 0
        self.eval_net.cuda()
        self.target_net.cuda()

    def choose_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()
            actions_value = self.eval_net.forward(state)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:
            action = random.randint(0, self.n_actions - 1)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_memory.push(state, action, reward, next_state, done)

    def learn(self, batch_size):
        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer.zero_grad()
        batch = self.replay_memory.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        batch_state = torch.stack(batch_state).cuda()
        batch_action = torch.LongTensor(batch_action).cuda()
        batch_reward = torch.FloatTensor(batch_reward).cuda()
        batch_next_state = torch.stack(batch_next_state).cuda()
        batch_done = torch.FloatTensor(batch_done).cuda()
        q_eval = self.eval_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + 0.99 * q_next.max(1)[0] * (1 - batch_done)
        loss = self.loss_func(q_eval, q_target)
        loss.backward()
        self.optimizer.step()

def train():
    dqn = DQN(4)
    writer = SummaryWriter('runs/rob538a2')

    for i_episode in tqdm(range(1000)):
        # initialize grid start agent at random position
        grid, goals = custom_grid(1)
        total_reward = 0
        # agent postion list of tuples
        # actions list of integers
        agent_pos = [(random.randint(0, 4), random.randint(0, 9))]
        state = render(agent_pos, [0], grid)
        state = torch.FloatTensor(state)
        state = state.permute(2, 0, 1)

        for t in range(10000):
            action = [dqn.choose_action(state, 0.1)]
            # pos, r, d, goals, neighbours, img
            agent_positions, reward, done, goals, neighbours, next_state = step(grid, action, agent_pos, goals)
            total_reward += reward[0]
            next_state = torch.FloatTensor(next_state)
            next_state = next_state.permute(2, 0, 1)
            dqn.store_transition(state, *action, *reward, next_state, done)
            if dqn.replay_memory.__len__() > 1000:
                dqn.learn(32)
            if done:
                break
            state = next_state
        writer.add_scalar('reward', total_reward, i_episode)
    # save model
    torch.save(dqn.eval_net.state_dict(), 'dqn.pth')
    writer.close()

if __name__ == '__main__':
    train()
