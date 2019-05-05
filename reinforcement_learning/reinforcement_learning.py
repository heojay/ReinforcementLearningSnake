#!/usr/bin/env python3

from reinforcement_learning.snake.prototype_snake import Snake
from random import randint, random
from abc import abstractmethod
from time import sleep
from os import system


class ReinforcementLearning:
    """
    Superclass for restricted learning algorithms
    Author: Adam Ross
    Date: 05/05/2019
    """

    DEFAULT_REWARD = 0  # default reward
    NEGATIVE_REWARD = -1  # the reward for when snake crashes
    POSITIVE_REWARD = 1  # the reward for when snake eats food
    EPISODES = 500  # number of episodes
    SPEED = 600  # the speed the snake moves
    SIZE = 3  # the length of the snake
    DIRS = {0: '\x1b[A', 1: '\x1b[B', 2: '\x1b[C', 3: '\x1b[D'}
    OPTIMAL_PATHS = 5  # the number of optimal paths displayed
    OPTIMAL_LEVELS = 5  # the number of consecutive levels paths displayed
    DISPLAY = False  # if the learning algorithm is displayed or not
    FOOD_POS = [2, 3]  # the food pos at initialization

    def __init__(self, eta, gamma, epsilon):
        """
        Class Initializer
        """
        self.eta = eta  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
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

    def choose_action(self):
        """
        Chooses an action in a state to update the q-value of
        """
        if random() < self.epsilon:
            self.action = randint(0, 3)
        else:
            self.action = self.rl_map[self.state[0]][self.state[1]]. \
                index(max(self.rl_map[self.state[0]][self.state[1]]))

    def play_episode(self):
        """
        Plays an episode until agent finds trophy or crashes
        """
        self.state = self.agent[0].copy()
        self.choose_action()
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

    def optimal_paths(self):
        """
        Finds and displays an optimal path determined by learning algorithm
        Starts from random position and draws a path to the food position
        """
        for k in range(self.OPTIMAL_PATHS):
            self.food = self.FOOD_POS
            self.agent = None

            for i in range(self.OPTIMAL_LEVELS):
                print(str(k + 1) + " : " + str(i + 1))
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
                        max_q_value = 0

                        for j in range(4):
                            if j not in actions:
                                max_q_value = \
                                    max([self.rl_map[self.agent[0][0]][self.
                                        agent[0][1]][j], max_q_value])

                        self.action = \
                            self.rl_map[self.agent[0][0]][self.agent[0][1]].\
                                                            index(max_q_value)
                        head_move = self.game.absolute_dirs(self.DIRS[self.
                                                            action])

                        if head_move:
                            self.agent[:0] = [head_move]
                            break
                        actions.append(self.action)
                self.game.snake = self.game.snake[1:]
                self.game.tail = self.agent[-1]
                self.game.display_snake()
