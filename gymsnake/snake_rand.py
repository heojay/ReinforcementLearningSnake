# Uses the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a random Discrete version

import gym
import gym_snake

MAX_TIMESTEPS = 100
FRAME_SPEED = 0.1

env = gym.make('snake-v0')
env.n_foods = 1
env.random_init = False
observation = env.reset()

for t in range(MAX_TIMESTEPS):
    env.render(frame_speed=FRAME_SPEED)
    print("timestep t={}".format(t))
    print(observation)
    action = env.action_space.sample() # take a random action
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
