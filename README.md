# 概述
本库记录强化学习算法的原理和实现。
已涵盖的算法包括：
* Deep Q-Network

# 算法
## DQN
算法参考DeepMind论文*Human-level control through deep reinforcement learning*，使用两个网络，其一为Behavior Network，其二为Target Network，两个网络在每个Episode后共享网络的权值。

### 建立网络
编写函数`build_model`，创建一个三层的全连接网络，使用Keras。前两层采用relu函数非线性化，最后一层线性输出。输入单元的维度为状态空间维度（4个维度），输出单元为两个维度（可以采取两种动作，0代表向左推，1代表向右推）。Hiden Layer设置24个节点。网络相当于一个Action-Value Function，输出在某个状态下采用不同Action的概率。
```python
def build_model(self):
    model = Sequential()
    model.add(Dense(24, input_dim=self.state_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(self.action_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    return model
```

### 策略
编写函数`get_action`，采用ε-greedy策略选择动作，即ε的概率随机选择动作，1-ε的概率选择贪心动作（价值函数最大的动作）。ε的值随着时间的推移而减小，表示exploration的倾向减小。
```python
def get_action(self, state):
    if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_size)
    else:
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])
```

### Experience Replay
DQN的一个创新之处在于采用了Experience Replay，即建立Replay Memory，每次训练从Memory中随机取mini-batch，消除训练样本间的相关性。我们采用`deque`即双向队列的数据结构，建立Memory，并规定Memory的最大长度（默认取2000）。

建立函数`append_sample`将状态动作序列写入Memory，写入的同时调整ε的值。
```python
def append_sample(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
```

### 目标网络权值更新
建立函数`update_target_model`，将目标网络的权值更新为与行为网络相同。
```python
def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())
```

### 网络训练
建立函数`train_model`，从Memory中取出样本并进行训练。注意，每次更新，行为网络都将向目标网络的方向移动。
```python
def train_model(self):
    if len(self.memory) < self.train_start:
        return
    batch_size = min(self.batch_size, len(self.memory))
    mini_batch = random.sample(self.memory, batch_size)
    update_input = np.zeros((batch_size, self.state_size))
    update_target = np.zeros((batch_size, self.state_size))
    action, reward, done = [], [], []

    for i in range(self.batch_size):
        update_input[i] = mini_batch[i][0]  # State
        action.append(mini_batch[i][1])  # 0 or 1
        reward.append(mini_batch[i][2])
        update_target[i] = mini_batch[i][3]  # Next state
        done.append(mini_batch[i][4])
    # values of behavior
    target = self.model.predict(update_input)
    # values of target
    target_val = self.target_model.predict(update_target)
    for i in range(self.batch_size):
        if done[i]:
            target[i][action[i]] = reward[i]
        else:
            target[i][action[i]] = reward[i] + self.gamma * (
                np.amax(target_val[i]))

    self.model.fit(update_input, target, batch_size=self.batch_size,
                   epochs=1, verbose=0)
```

### 主程序
使用`CartPole-v1`训练环境，设置最大训练Episode数。每一个Episode，重置训练环境。在没有到达终止状态（失去平衡）时，选择并执行动作，将(S, A, R, S')序列写入Memory，并训练模型。当到达终止状态时，更新目标网络，输出Episode的基本信息。每50个Episode存储一次网络权重。
```python
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env._max_episode_steps = None
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100
            agent.append_sample(state, action, reward, next_state, done)
            agent.train_model()
            score += reward
            state = next_state

            if done:
                agent.update_target_model()
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
                if np.mean(scores[-min(10, len(scores)):]) > 1000:
                    sys.exit()

        if e % 50 == 0:
            agent.model.save_weights("./save/cartpole_dqn.h5")
```
