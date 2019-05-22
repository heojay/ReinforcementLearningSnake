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
    Date: 22/05/2019
    """

    ETA = 0.1  # learning rate
    GAMMA = 0.95  # discount factor
    EPSILON = 0.9  # exploration rate
    DEFAULT_REWARD = 0  # default reward
    NEGATIVE_REWARD = -1  # the reward for when snake crashes
    POSITIVE_REWARD = 1  # the reward for when snake eats food
    SPEED = 600  # the speed the snake moves
    DIRS = {0: '\x1b[A', 1: '\x1b[B', 2: '\x1b[C', 3: '\x1b[D'}
    OPTIMAL_PATHS = 121  # the number of optimal paths displayed
    DISPLAY = False  # if the learning algorithm is displayed or not
    TROPHY_POS = [5, 5]   # trophy pos at init
    FIXED_TROPHY = True  # if the trophy position is fixed at every episode
    PURE_RANDOM = False  # if the trophy is pure random at every episode
    GROW = True  # if the agent (snake) increments at tail when gets a trophy
    FIXED_AGENT = False  # if the agent has a fixed starting state each episode
    FIXED_AGENT_POS = [3, 8]  # the pos of the agent when FIXED_AGENT is True
    PATH = 'reinforcement_learning/data/'  # the path to table data file

    def __init__(self, file_name, episodes, levels, size):
        """
        Class Initializer
        :param file_name: the name of the file the learning data is saved to
        :param episodes: the number of learning episodes
        :param levels: the number of snake game levels learned and played back
        :param size: the size of the agent at learning initialization
        """
        self.file, self.reward = self.PATH + file_name, self.DEFAULT_REWARD
        self.episodes, self.levels, self.size = episodes, levels, size
        self.game = Snake(True)  # snake game instance
        self.rl_map_levels = {i: None for i in range(self.levels)}
        self.rl_map = [[[0] * 4 for _ in range(self.game.N)]
                       for _ in range(self.game.N)]  # matrix of states
        self.state, self.trophy_count, self.trophy = [], [], self.TROPHY_POS
        self.agent = self.action = None  # the agent

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

        if self.PURE_RANDOM:
            self.trophy = [randint(0, self.game.N - 1),
                           randint(0, self.game.N - 1)]
        elif not self.FIXED_TROPHY:
            self.trophy = self.game.new_food()

    def init_snake(self):
        """
        Initializes the snake and game environment
        """
        self.game = Snake(True)
        self.game.restricted_learning = True
        self.game.set_food(self.trophy)
        self.trophy = self.game.food

        if not self.agent:
            self.game.set_snake()

            if self.FIXED_AGENT:
                self.game.snake = self.FIXED_AGENT_POS
            self.agent = self.game.snake

            if len(self.agent) > self.size:
                self.agent.pop()

            while len(self.agent) < self.size:
                self.game.add_tail(True)

    def learn(self):
        """
        Runs the Q-learning algorithm for number of given episodes
        """
        for j in range(1, self.levels + 1):
            with open(self.file[:-5] + str(self.size - 1) + ".txt", 'rb') as h:
                self.rl_map_levels = loads(h.read())
            self.rl_map, count_trophies = self.rl_map_levels[0], 0

            if j > 1:
                if self.GROW:
                    self.size += 1

                if self.FIXED_TROPHY:
                    self.trophy = self.game.new_food()

            for i in range(1, self.episodes + 1):
                if self.DISPLAY:
                    system('clear')
                    print("Level " + str(j) + " of " + str(self.levels) +
                          ": Episode " + str(i) + " of " + str(self.episodes))
                self.agent = None
                self.init_snake()
                self.game.snake_crashed = False
                self.play_episode()

                if not self.game.snake_crashed:
                    count_trophies += 1

                if self.levels == 1:
                    if i % (self.episodes / 10) == 0:
                        self.trophy_count.append(count_trophies)
                        count_trophies = 0

            if self.levels > 1:
                self.trophy_count.append(count_trophies)
            self.rl_map_levels[0] = self.rl_map

            with open(self.file, 'wb') as file:
                dump(self.rl_map_levels, file)
        return self.trophy_count

    def optimal_paths(self):
        """
        Finds and displays an optimal path determined by learning algorithm
        Starts from random position and draws a path to the food position
        """
        with open(self.file, 'rb') as handle:
            self.rl_map_levels = loads(handle.read())

        if self.OPTIMAL_PATHS == self.game.N * self.game.N:
            paths = [[i, j] for i in range(self.game.N)
                     for j in range(self.game.N)]

        for k in range(self.OPTIMAL_PATHS):
            self.trophy, self.agent, count = [5, 5], None, 0

            for i in range(self.levels):
                self.rl_map = self.rl_map_levels[i]

                if self.DISPLAY:
                    print("Optimal path " + str(k + 1) + " -> level " +
                          str(i + 1))

                if self.OPTIMAL_PATHS == self.game.N * self.game.N:
                    self.agent = [paths[k]]
                else:
                    self.agent = [self.trophy]

                if i > 0:
                    self.agent += \
                        self.game.snake[:(self.size - self.levels) + i]
                    self.trophy = self.game.new_food()
                self.init_snake()
                self.agent = self.game.snake = self.agent

                while self.agent[0] != self.trophy:
                    actions = [4]

                    if self.agent[0] in self.agent[1:]:
                        count = -5
                        break

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

                        if head_move and (0 <= head_move[0] < self.game.N) and\
                                (0 <= head_move[1] < self.game.N):
                            self.agent[:0] = [head_move]
                            count += 1
                            break
                        actions.append(self.action)

                if len(self.agent) > 1:
                    self.game.snake = self.game.snake[1:]

                if len(self.agent) > 0 and self.DISPLAY:
                    self.game.tail = self.agent[-1]
                    self.game.display_snake()

            if self.OPTIMAL_PATHS == self.game.N * self.game.N:
                paths[k] = [count]
        return [i[0] for i in paths]
