#!/usr/bin/env python3

from os import system
from time import sleep
from random import randint, random
from prototype_snake import Snake


class LearningQ:
    """
    Prototype for the q-learning algorithm
    Author: Adam Ross
    Date: 01/05/2019
    """

    ETA = 0.1  # learning rate
    GAMMA = 0.95  # discount
    EPSILON = 1  # exploration
    DEFAULT_REWARD = 0  # default reward
    NEGATIVE_REWARD = -1  # the reward for when snake crashes
    POSITIVE_REWARD = 1  # the reward fro when snake eats food
    EPISODES = 500  # number of episodes
    SPEED = 600  # the speed the snake moves
    SIZE = 1  # the length of the snake
    DIRS = {0: '\x1b[A', 1: '\x1b[B', 2: '\x1b[C', 3: '\x1b[D'}
    DISPLAY = False

    def __init__(self):
        """
        Class Initializer
        """
        self.game = Snake(True)  # snake game instance
        self.ql_map = [[[0] * 4 for _ in range(self.game.N)]
                       for _ in range(self.game.N)]  # matrix of states
        self.agent = None  # the agent
        self.trophy = self.game.food  # the trophy with reward = 1
        self.state = []  # the state the action q-value is being updated
        self.action = None  # the agent action; direction from the state
        self.reward = self.DEFAULT_REWARD

    def update_state(self):
        """
        Updates a state action q-value
        """
        if self.state != self.agent[0]:
            if self.game.snake_crashed:
                self.reward, nxt_state = self.NEGATIVE_REWARD, [0]
            elif self.agent[0] == self.trophy:
                self.reward, nxt_state = self.POSITIVE_REWARD, [0]
            else:
                nxt_state = self.ql_map[self.agent[0][0]][self.agent[0][1]].\
                    copy()
            state = self.ql_map[self.state[0]][self.state[1]].copy()
            self.ql_map[self.state[0]][self.state[1]][self.action] += \
                self.ETA * (self.reward + self.GAMMA * max(nxt_state) -
                            state[self.action])
            self.reward = self.DEFAULT_REWARD

    def play_episode(self):
        """
        Plays an episode until agent finds trophy or
        """
        self.state = self.agent[0].copy()

        if random() < self.EPSILON:
            self.action = randint(0, 3)
        else:
            self.action = self.ql_map[self.state[0]][self.state[1]].\
                index(max(self.ql_map[self.state[0]][self.state[1]]))
        self.game.move_snake(self.game.absolute_dirs(self.DIRS[self.action]))
        self.update_state()

        if self.DISPLAY:
            self.game.display_snake()

        if not self.game.snake_crashed and not self.agent[0] == self.trophy:
            sleep(1 / self.SPEED)
            self.play_episode()

    def init_snake(self):
        """
        Initializes the snake and game environment
        """
        self.game = Snake(True)
        self.agent = self.game.snake

        if len(self.agent) > self.SIZE:
            self.agent.pop()

        while len(self.agent) < self.SIZE:
            self.game.add_tail(True)

            for j in range(2):
                if not 0 <= self.agent[-1][j] < self.game.N:
                    self.agent.pop()
                    break

    def q_learn(self):
        """
        Runs the Q-learning algorithm for number of given episodes
        """
        for i in range(self.EPISODES):
            system('clear')
            print("Episode " + str(i + 1) + " out of " + str(self.EPISODES))
            self.init_snake()
            self.game.snake_crashed = False
            self.play_episode()

    def optimal_paths(self):
        """
        Finds and displays an optimal path determined by q-learning results
        Starts from random position and creates a path to the food position
        """
        self.init_snake()

        while self.agent[0] != self.trophy:
            self.action = self.ql_map[self.agent[0][0]][self.agent[0][1]].\
                index(max(self.ql_map[self.agent[0][0]][self.agent[0][1]]))
            head_move = self.game.absolute_dirs(self.DIRS[self.action])

            if head_move:
                self.agent[:0] = [head_move]
        self.agent.pop(0)
        self.game.display_snake()


if __name__ == "__main__":
    app = LearningQ()
    app.q_learn()
    app.optimal_paths()
