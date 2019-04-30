# Uses the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a random Discrete version

import gym
import gym_snake

env = gym.make('snake-v0')
env.n_foods = 1
env.random_init = False
observation = env.reset()

for t in range(100):
    env.render()
    print("interval t={}".format(t))
    print(observation)
    action = env.action_space.sample() # take a random action
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
