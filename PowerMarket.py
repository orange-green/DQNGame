import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = torch.relu(self.fc1(hn[-1]))
        return self.fc2(x)


class UserAgent:
    def __init__(self, user_id, initial_demand):
        self.user_id = user_id
        self.demand = initial_demand
        self.price_sensitivity = random.uniform(0.1, 1.0)
        self.remaining_energy = 100
        self.total_revenue = 0
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.model = DQN(input_dim=4, output_dim=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_strategy(self, market_price, neighbor_strategies):
        state = np.array([market_price, np.mean(neighbor_strategies), self.remaining_energy, self.demand])
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # 维度: [1, 1, 4]

        if random.random() < self.exploration_rate:
            action = random.choice([0, 1, 2, 3])
        else:
            action_values = self.model(state_tensor)
            action = torch.argmax(action_values).item()

        if action == 0:
            sale_price = market_price * (1 - self.price_sensitivity)
        elif action == 1:
            sale_price = market_price
        elif action == 2:
            sale_price = market_price * (1 + self.price_sensitivity)
        else:
            sale_price = None

        if sale_price is not None:
            self.total_revenue += sale_price * self.remaining_energy
            self.remaining_energy = 0

        reward = self.calculate_reward(False)
        self.memory.append((state, action, reward))

        return sale_price

    def calculate_reward(self, is_year_end):
        reward = self.total_revenue
        if is_year_end and self.remaining_energy > 0:
            penalty = self.remaining_energy * 0.1
            reward -= penalty
        return reward

    def learn(self, is_year_end):
        if len(self.memory) < 32:
            return

        minibatch = random.sample(self.memory, 32)
        for state, action, _ in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # 维度: [1, 1, 4]
            reward = self.calculate_reward(is_year_end)
            target = reward + self.discount_factor * torch.max(self.model(state_tensor)).item()

            target_f = self.model(state_tensor).detach()  # 获取当前的Q值，维度: [1, 4]
            target_f = target_f.view(-1)  # 调整为一维张量
            target_f[action] = target  # 更新目标Q值，确保目标维度为 [4]

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor).view(-1), target_f)  # 确保目标和输入的维度一致
            loss.backward()
            self.optimizer.step()

        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

    def receive_messages(self, neighbor_strategies):
        if neighbor_strategies:
            avg_neighbor_strategy = np.mean(neighbor_strategies)
            self.price_sensitivity = (self.price_sensitivity + avg_neighbor_strategy) / 2

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))


class Market:
    def __init__(self, num_users):
        self.users = [UserAgent(i, random.randint(10, 50)) for i in range(num_users)]
        self.market_price = 100
        self.historical_prices = []
        self.total_revenues = []

    def dynamic_price_function(self, total_demand, total_supply):
        if total_supply > total_demand:
            return max(50, self.market_price - (total_supply - total_demand) / 10)
        else:
            return min(150, self.market_price + (total_demand - total_supply) / 10)

    def simulate(self):
        for step in range(100):
            is_year_end = step == 99
            total_demand = sum(user.demand for user in self.users)
            total_supply = sum(user.remaining_energy for user in self.users)

            self.market_price = self.dynamic_price_function(total_demand, total_supply)
            self.historical_prices.append(self.market_price)

            neighbor_strategies = [user.remaining_energy for user in self.users]
            for user in self.users:
                user.receive_messages(neighbor_strategies)
                user.update_strategy(self.market_price, neighbor_strategies)

            total_revenue = sum(user.total_revenue for user in self.users)
            self.total_revenues.append(total_revenue)

            for user in self.users:
                user.learn(is_year_end)

            print(f"Step {step}: Market Price = {self.market_price}, Total Revenue = {total_revenue}")

    def plot_results(self):
        plt.figure(figsize=(14, 6))

        # 市场价格变化图
        plt.subplot(1, 2, 1)
        plt.plot(self.historical_prices, label="Market Price")
        plt.title("Market Price Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Market Price")
        plt.legend()

        # 用户总收益变化图
        plt.subplot(1, 2, 2)
        plt.plot(self.total_revenues, label="Total Revenue", color="orange")
        plt.title("Total Revenue Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Total Revenue")
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig(f'figs/{time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))}.png')


if __name__ == "__main__":
    # 示例运行
    market = Market(num_users=1000)
    market.simulate()

    # 保存模型
    for user in market.users:
        user.save_model(f"./userModel/user_{user.user_id}_model.pth")

    # 可视化结果
    market.plot_results()
