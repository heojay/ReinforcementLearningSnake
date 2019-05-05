#!/usr/bin/env python3

from reinforcement_learning.snake.prototype_snake import Snake
from random import randint, random
from abc import abstractmethod
from pickle import loads, dump
from time import sleep
from os import system
from cmath import inf


class ReinforcementLearning:
    """
    Superclass for restricted learning algorithms
    Author: Adam Ross
    Date: 05/05/2019
    """

    ETA = 0.1  # learning rate
    GAMMA = 0.95  # discount factor
    EPSILON = 0.9  # exploration rate
    DEFAULT_REWARD = 0  # default reward
    NEGATIVE_REWARD = -1  # the reward for when snake crashes
    POSITIVE_REWARD = 1  # the reward for when snake eats food
    EPISODES = 1000  # number of episodes
    SPEED = 600  # the speed the snake moves
    SIZE = 1  # the length of the snake
    DIRS = {0: '\x1b[A', 1: '\x1b[B', 2: '\x1b[C', 3: '\x1b[D'}
    OPTIMAL_PATHS = 5  # the number of optimal paths displayed
    OPTIMAL_LEVELS = 5  # the number of consecutive levels paths displayed
    DISPLAY = False  # if the learning algorithm is displayed or not
    FOOD_POS = [2, 3]  # the food pos at initialization

    def __init__(self, file_name):
        """
        Class Initializer
        :param file_name: the name of the file the learning data is saved to
        """
        self.file = 'reinforcement_learning/data/' + file_name
        self.game = Snake(True)  # snake game instance
        self.rl_map_levels = {i: None for i in range(self.OPTIMAL_LEVELS)}
        self.rl_map = [[[0] * 4 for _ in range(self.game.N)]
                       for _ in range(self.game.N)]  # matrix of states
        self.agent = None  # the agent
        self.trophy = None  # the trophy with reward = 1
        self.state = []  # the state the action q-value is being updated
        self.action = None  # the agent action; direction from the state
        self.food = self.FOOD_POS
        self.reward = self.DEFAULT_REWARD

    @abstractmethod
    def update_state(self):
        """
        Updates a state action q-value using an update rule
        """

    def update_rule(self, state, action, next_action):
        """
        Update rule for SARSA and Q-learning algorithms
        :param state: the state being updated
        :param action: the action in the state being updated
        :param next_action: the action in the next state
        """
        self.rl_map[state[0]][state[1]][action] += \
            self.ETA * (self.reward + self.GAMMA * next_action -
                        self.rl_map[state[0]][state[1]][action])

    def choose_action(self):
        """
        Chooses an action in a state to update the q-value of
        """
        if random() < self.EPSILON:
            self.action = randint(0, 3)
        else:
            self.action = self.rl_map[self.state[0]][self.state[1]]. \
                index(max(self.rl_map[self.state[0]][self.state[1]]))

    def play_episode(self):
        """
        Plays an episode until agent finds trophy or crashes
        """
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
        self.game.restricted_learning = True
        self.game.set_food(self.food)
        self.trophy = self.game.food

        if not self.agent:
            self.game.set_snake()
            self.agent = self.game.snake

            if len(self.agent) > self.SIZE:
                self.agent.pop()

            while len(self.agent) < self.SIZE:
                self.game.add_tail(True)

    def learn(self):
        """
        Runs the Q-learning algorithm for number of given episodes
        """
        for j in range(1, self.OPTIMAL_LEVELS + 1):
            for i in range(1, self.EPISODES + 1):
                system('clear')
                print("Level " + str(j) + " of " + str(self.OPTIMAL_LEVELS)
                      + ": Episode " + str(i) + " of " + str(self.EPISODES))
                self.agent = None
                self.init_snake()
                self.game.snake_crashed = False
                self.play_episode()
            self.food = self.game.new_food()
            self.rl_map_levels[j - 1] = self.rl_map
            self.rl_map = [[[0] * 4 for _ in range(self.game.N)]
                           for _ in range(self.game.N)]
        with open(self.file, 'wb') as file:
            dump(self.rl_map_levels, file)

    def optimal_paths(self):
        """
        Finds and displays an optimal path determined by learning algorithm
        Starts from random position and draws a path to the food position
        """
        with open(self.file, 'rb') as handle:
            self.rl_map_levels = loads(handle.read())

        for k in range(self.OPTIMAL_PATHS):
            self.food = self.FOOD_POS
            self.agent = None

            for i in range(self.OPTIMAL_LEVELS):
                print("Optimal path " + str(k + 1) + " -> level " + str(i + 1))
                self.rl_map = self.rl_map_levels[i]

                if i > 0:
                    self.agent = [self.trophy] + \
                                       self.game.snake[:self.SIZE + i - 1]
                    self.food = self.game.new_food()
                self.init_snake()
                self.game.snake = self.agent
                self.agent = self.game.snake

                while self.agent[0] != self.trophy:
                    actions = [4]

                    while True:
                        max_q_value = -inf

                        for j in range(4):
                            if j not in actions:
                                new_value = \
                                    max([self.rl_map[self.agent[0][0]][self.
                                        agent[0][1]][j], max_q_value])

                                if new_value > max_q_value:
                                    max_q_value, self.action = new_value, j
                        head_move = self.game.absolute_dirs(self.DIRS[self.
                                                            action])

                        if head_move:
                            self.agent[:0] = [head_move]
                            break
                        actions.append(self.action)
                self.game.snake = self.game.snake[1:]
                self.game.tail = self.agent[-1]
                self.game.display_snake()
