import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 参数设置
num_steps = 365  # 一年365天
initial_energy = 100  # 初始电力
num_states = 100  # 状态的离散化数量
num_actions = 2  # 出售或不出售
alpha = 0.001  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_min = 0.1  # 最小探索率
epsilon_decay = 0.995  # 探索率衰减
batch_size = 32  # 批次大小
memory = deque(maxlen=2000)  # 经验回放记忆


# 深度 Q 网络
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 状态离散化函数
def discretize(price):
    return min(int(price), num_states - 1)


# 市场价格生成函数
def market_price_generator():
    return np.random.uniform(50, 150)


# 检查点保存函数
def save_checkpoint(model, optimizer, episode, filename="checkpoint.pth"):
    torch.save(
        {
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


# 检查点加载函数
def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["episode"]


# 测试模型效果的方法
def test_model(model, num_tests=10):
    total_reward = 0
    for _ in range(num_tests):
        energy = initial_energy
        for t in range(num_steps):
            market_price = market_price_generator()
            state = np.array([discretize(market_price), energy])
            action = np.argmax(model(torch.FloatTensor(state)).detach().numpy())

            if action == 1:  # 出售
                reward = market_price * 10 - 5
                energy -= 10
            else:  # 不出售
                reward = -20

            if t == num_steps - 1 and energy > 0:
                reward -= 20

            total_reward += reward
    average_reward = total_reward / num_tests
    print("平均年度总收益:", average_reward)


# 训练模型的方法
def train_model(model, optimizer, num_episodes=1000):
    global epsilon  # 声明使用全局变量 epsilon
    for episode in range(num_episodes):
        energy = initial_energy
        total_reward = 0

        for t in range(num_steps):
            market_price = market_price_generator()
            state = np.array([discretize(market_price), energy])

            # epsilon-greedy 策略选择动作
            if np.random.rand() <= epsilon:
                action = random.choice([0, 1])  # 随机选择
            else:
                action = np.argmax(model(torch.FloatTensor(state)).detach().numpy())  # 选择 Q 值最大的动作

            # 计算奖励
            if action == 1:  # 出售
                reward = market_price * 10 - 5
                energy -= 10
            else:  # 不出售
                reward = -20

            # 年末清场操作
            if t == num_steps - 1 and energy > 0:
                reward -= 20

            total_reward += reward

            # 存储经验
            memory.append((state, action, reward, state, False))

            # 经验回放
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, s_next, done in minibatch:
                    target = r + (gamma * np.max(model(torch.FloatTensor(s_next)).detach().numpy())) * (not done)
                    target_f = model(torch.FloatTensor(s)).detach()
                    target_f[a] = target
                    optimizer.zero_grad()
                    loss = criterion(model(torch.FloatTensor(s)), target_f)
                    loss.backward()
                    optimizer.step()

        # 逐渐减少探索率
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # 每100个回合保存一次检查点
        if episode % 50 == 0:
            save_checkpoint(model, optimizer, episode)


if __name__ == "__main__":
    # 初始化 DQN
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    criterion = nn.MSELoss()

    # 训练模型
    train_model(model, optimizer, 100)

    # 测试训练后的策略
    test_model(model)
