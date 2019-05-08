import random
import pygame
import matplotlib.pyplot as plt

from snake1 import Env

env = Env()
state_size = 2
action_size = 4
episodes = 300

scores = []

total = 0

for i in range(episodes):

    observation, reward, done, info = env.reset()
    total = 0

    while not done:

        observation, reward, done, info = env.shortest_step()
        if (reward == 1):
            total += 1

        if done:
            scores.append(total)

average = sum(scores) / episodes

print(average)

fig = plt.figure()
plt.plot(scores)
fig.suptitle('shortcut')
plt.xlabel('episode')
plt.ylabel('average fruit')
plt.show()