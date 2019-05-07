# Uses the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a blind version Q-learning version
# Note that this only works for one food goal

import gym
import gym_snake
import numpy as np
import random

class SnakeQlearning:
    """
    Snake game that uses Q-learning
    Author: Johan Alfredeen
    Date: 2019-05-06
    """

    ETA = 0.1  # learning rate
    GAMMA = 0.95  # discount
    EPSILON = 0.2  # exploration: greedy - random
    REWARD = 0  # default reward
    EPISODES = 20  # number of episodes
    MAX_EPISODE_INTERVALS = 100 # max number of intervals per episode

    def __init__(self):
        """
        Class Initializer
        """
        self.env = gym.make('snake-v0')
        self.env.n_foods = 1
        self.env.random_init = False
        self.env.grid_size = [6,6]
        self.env.snake_size = 2
        self.frame_speed = 5
        self.snake = None
        self.trophy = None
        self.width = self.env.grid_size[0]  # 15
        self.height = self.env.grid_size[1] # 15
        self.q = Qlearn(self.width, self.height, self.ETA, self.GAMMA, self.EPSILON, self.REWARD)


    def select_action(self):
        """
        Select an action with a probability of EPSILON of choosing max Q-value
        :return: an action index
        """
        if random.randint(0,100) < self.EPSILON * 100:
            # Explore
            # select a random direction including the max
            idx = random.randint(0,3) 
        else:
            # use the direction of the max-value
            idx = self.q.get_index_max_value(self.snake.head[0], self.snake.head[1])
        return self.convert_idx_to_action(idx)

    def convert_idx_to_action(self, idx):
        if idx == 0:
            return self.snake.UP
        elif idx == 1:
            return self.snake.RIGHT
        elif idx == 2:
            return self.snake.DOWN
        elif idx == 3:
            return self.snake.LEFT

    def convert_action_to_index(self, action):
        if action == self.snake.UP:
            return 0
        elif action == self.snake.RIGHT:
            return 1
        elif action == self.snake.DOWN:
            return 2
        elif action == self.snake.LEFT:
            return 3

    def is_legal_action(self, action):
        """
        Checks if a given action is legal based on direction of snake
        :return: True or False
        """
        dir = self.snake.direction
        if action == self.snake.RIGHT and dir == self.snake.LEFT:
            return False
        elif action == self.snake.LEFT and dir == self.snake.RIGHT:
            return False
        elif action == self.snake.UP and dir == self.snake.DOWN:
            return False
        elif action == self.snake.DOWN and dir == self.snake.UP:
            return False
        else:
            return True

    def display_qvalues(self):
        print(self.q.qmap)

    def play(self):
        for _ in range(self.EPISODES):
            observation = self.env.reset()  # observation contains color
            game_controller = self.env.controller
            self.snake = game_controller.snakes[0]
            state = self.snake.head # state is coord location of snake
            print("state:{0}".format(state))
            totalreward = 0

            for t in range(self.MAX_EPISODE_INTERVALS):
                self.env.render(self.frame_speed)
                print("interval t={}".format(t))
                action = self.select_action() # select action from Q matrix
                if (action == None):
                    break
                observation, reward, done, _ = self.env.step(action)
                print("reward:{0}, done:{1}".format(reward, done))
                nxt_state = self.snake.head
                print("next state:{0}".format(nxt_state))
                totalreward += reward
                action_idx = self.convert_action_to_index(action)
                # TODO: what should happen if we go out of bounds?
                if state[0] < self.width and state[1] < self.height:
                    self.q.update_state(reward, state, action_idx, nxt_state) # update the Q matrix
                state = nxt_state
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                if state[0] >= self.width or state[1] >= self.height:
                    # why is this needed?
                    break
                else:
                    # We finish the episode if we found the fruit
                    if game_controller.grid.food_space(self.snake.head):
                        # TODO: must solve the moving fruit!
                        print("FOUND THE FRUIT")
                        break

            print("Finished episode with total accumulated reward = {0}".format(totalreward))
        self.env.close()



class Qlearn:
    """
    Implementation of Q-learning
    Holds Q-values in a matrix with tuples of 4 values
    Date: 2019-05-06
    """

    lr = 0.1 # alpha learn_rate
    disc = 0.95  # gamma discount
    expl = 1.0 # exploration
    reward = 0 # default_reward

    def __init__(self, width, height, lr, disc, expl, reward):
        self.w = width
        self.h = height
        self.lr = lr
        self.disc = disc
        self.expl = expl
        self.reward = reward
        self.qmap = [[[0] * 4 for _ in range(height)]
                       for _ in range(width)]  # matrix of states: N E S W, init to 0 values

    def get_index_max_value(self, row, col):
        """
        :return: the index (N E S W) of the max value of the state cell
        """
        values=self.qmap[col][row]
        return values.index(max(values))

    def update_state(self, reward, state, action_idx, nxt_state):
        """
        Updates the Q-value of a state
        """
        self.qmap[state[0]][state[1]][action_idx] = self.get_update_qvalue(reward, state, action_idx, nxt_state)

    def get_update_qvalue(self, reward, state, action_idx, nxt_state):
        """
        Calculates a new Q-value for a state
        :return: the new Q-value
        """
        oldval = self.qmap[state[0]][state[1]][action_idx]
        learnedval = (reward + self.disc * max(nxt_state) )
        qv = (1-self.lr) * oldval + self.lr * learnedval
        print("Q-value={0}", qv)
        return qv

    def get_optimal_path(self):
        """
        Determine the optimal path per game episode
        :return: an ordered list of states
        """
        # TODO:
        p=[]
        #while True:
        return p





if __name__ == "__main__":
    app = SnakeQlearning()
    app.play()
    print("Q-values")
    app.display_qvalues()
    #print("Optimal path")
    #path = app.q.get_optimal_path()

