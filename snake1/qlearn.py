import random
import pygame
import matplotlib.pyplot as plt

from snake1 import Env

class Agent:
    def __init__(self, env, Qtable, epsilon=1.0, alpha=0.5, gamma=0.9):
        self.env = env
        self.Qtable = Qtable
        self.actions = list(range(self.env.action_size))

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_maxQ(self, state):
        if state not in self.Qtable:
            self.Qtable[state] = dict((action, 0.0) for action in self.actions)
        return max(self.Qtable[state].values())

    def choose_action(self, state):
        if state not in self.Qtable:
            self.Qtable[state] = dict((action, 0.0) for action in self.actions)
        if random.random() > self.epsilon:
            maxQ = self.get_maxQ(state)
            action = random.choice([k for k in self.Qtable[state].keys() if self.Qtable[state][k] == maxQ])
        else:
            action = random.choice(self.actions)
        return action

    def learn(self, state, action, reward, next_state):
        self.Qtable[state][action] += self.alpha * (reward+ (self.gamma * self.get_maxQ(next_state)) - self.Qtable[state][action])


env = Env()
state_size = 2
action_size = 4
episodes = 30000

scores = []
Q = dict()

agent = Agent(env=env, Qtable = Q, epsilon=0.3, alpha=0.01, gamma=0.01)

for e in range(episodes):
    done = False
    total = 0
    state, reward, done, info = env.reset()

    while not done:
        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state)

        if (reward == 1):
            total += 1
        state = next_state

        if done:
            scores.append(total)

average_scores = []

for i in range(episodes//100):
    average_scores.append(sum(scores[(i-1)*100:i*100]) / 100)

fig = plt.figure()
plt.plot(average_scores)
fig.suptitle('Q-Learning')
plt.xlabel('episode')
plt.ylabel('average fruit')
plt.show()

