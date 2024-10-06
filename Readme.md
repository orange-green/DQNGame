用于模拟电力市场的用户博弈情景。

参考思路：[Farama-Foundation/MAgent2: An engine for high performance multi-agent environments with very large numbers of agents, along with a set of reference environments (github.com)](https://github.com/Farama-Foundation/MAgent2)

项目代码结构

1. **深度 Q 网络（DQN）**：

   - 使用 PyTorch 构建深度 Q 网络，使用 LSTM 处理时间序列数据，以便模型能够记忆更长时间的信息。输入维度设置为 4（即市场价格、邻居策略均值、剩余电力、需求）。

   - 状态包括市场价格、邻居策略的平均值、用户的策略、用户需求和剩余电力。

2. **经验回放**：

   - 使用 `deque` 存储用户的经验，随机抽样进行学习，打破数据之间的相关性，提高学习效果。

3. **动态策略更新**：

   - 用户在选择策略时，考虑了市场价格、邻居的平均策略、自己的出价、需求，这形成了一个更复杂的状态表示。

4. **优化器**：

   - 使用 Adam 优化器来更新模型参数，提升训练效率。