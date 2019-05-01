# Uses the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a blind version Q-learning version
# Note that this only works for one food goal

# TODO: look into GoalEnv

import gym
import gym_snake
import numpy as np
import random

class SnakeQlearning:

    ETA = 0.1  # learning rate
    GAMMA = 0.95  # discount
    EPSILON = 1  # exploration: greedy - random
    REWARD = 0  # default reward
    EPISODES = 1  # number of episodes
    TIME_INTERVALS = 10 # max number of time intervals per episode

    def __init__(self):
        """
        Class Initializer
        """
        self.env = gym.make('snake-v0')
        self.env.n_foods = 1
        self.env.random_init = False
        self.snake = None
        width = self.env.grid_size[0]
        height = self.env.grid_size[1]
        self.q = Qlearn(width, height, self.ETA, self.GAMMA, self.EPSILON, self.REWARD)

        ##env.reset()
        ##game_controller = env.controller
        ##print(game_controller.snakes[0])

    def select_action(self):
        """
        TODO: select an action with a probability of EPSILON of choosing max Q-value
        :return: an action index
        """
        snake_dir = self.snake.direction
        snakex = self.snake.head[0]
        snakey = self.snake.head[1]
        # Begin with totaly random movement
        while True:
            action = self.env.action_space.sample() # take a random action
            if action == self.snake.RIGHT and snake_dir != self.snake.LEFT:
                return self.snake.RIGHT
            elif action == self.snake.LEFT and snake_dir != self.snake.RIGHT:
                return self.snake.LEFT
            elif action == self.snake.UP and snake_dir != self.snake.DOWN:
                return self.snake.UP
            elif action == self.snake.DOWN and snake_dir != self.snake.UP:
                return self.snake.DOWN
            else:
                # Loop to try another action
                pass

    def selectNextAction(self):
        rand1 = random.randint(0,100)
        if rand1 < self.EPSILON * 100:
            # use max-value
            qc = self.q.getMaxValue(self.snake.head[0], self.snake.head[1])
            if qc.maxdir == "north":
                return self.snake.UP
            elif qc.maxdir == "south":
                return self.snake.DOWN
            elif qc.maxdir == "west":
                return self.snake.LEFT
            elif qc.maxdir == "east":
                return self.snake.RIGHT
            else:
                return None
        else:
            # select a random q-value incl max-value
            return random.randint(0,3) 

    def displayQvalues(self):
        self.q.displayMatrixes()

    def play(self):
        for i_episode in range(self.EPISODES):
            observation = self.env.reset()
            
            game_controller = self.env.controller
            snakes_array = game_controller.snakes
            self.snake = snakes_array[0]
            grid_object = game_controller.grid
            #coord_food = get_food_location(grid_object, game_controller)
            #print("food located at {}".format(coord_food))

            prev_action = 0
            for t in range(self.TIME_INTERVALS):
                self.env.render()
                print("interval t={}".format(t))
                #print(observation)
                action = self.select_action()
                #prev_action = action
                if (action == None):
                    break
                observation, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

        self.env.close()



class Qlearn:
    lr = 0.1 # alpha learn_rate
    disc = 0.95  # gamma discunt
    expl = 1.0 # exploration
    reward = 0 # default_reward

    def __init__(self, width, height, lr, disc, expl, reward):
        self.w = width
        self.h = height
        self.lr = lr
        self.disc = disc
        self.expl = expl
        self.reward = reward
        self.mtx_north = np.random.rand(height, width)
        self.mtx_south = np.random.rand(height, width)
        self.mtx_east = np.random.rand(height, width)
        self.mtx_west = np.random.rand(height, width)

    def getUpdateQValue(self, row, col, oldval):
        cellmax = self.getMaxValue(row, col)
        learnedval = (self.reward + self.disc * cellmax.maxval )
        qv = (1-self.lr) * oldval + self.lr * learnedval
        return qv

    def getMaxValue(self, row, col):
        v = self.mtx_north[row, col]
        qc = Qcell(row, col, v, "north")
        if self.mtx_south[row, col] > v:
            v = self.mtx_south[row, col]
            qc = Qcell(row, col, v, "south")
        if self.mtx_east[row, col] > v:
            v = self.mtx_east[row, col]
            qc = Qcell(row, col, v, "east")
        if self.mtx_west[row, col] > v:
            v = self.mtx_west[row, col]
            qc = Qcell(row, col, v, "west")
        return qc

    def getQValueNorth(self, row, col):
        return self.mtx_north[row, col]

    def getQValueSouth(self, row, col):
        return self.mtx_south[row, col]

    def getQValueEast(self, row, col):
        return self.mtx_east[row, col]

    def getQValueWest(self, row, col):
        return self.mtx_west[row, col]

    def displayMatrixes(self):
        print("matrixes")
        print(self.mtx_north)


class Qcell:
    def __init__(self, row, col, maxval, maxdir):
        self.row = row
        self.col = col
        self.maxval = maxval
        self.maxdir = maxdir




if __name__ == "__main__":
    app = SnakeQlearning()
    app.play()
    print("Q-values")
    app.displayQvalues()


