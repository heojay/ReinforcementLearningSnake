#!/usr/bin/env python3

from time import sleep
from random import randint
from prototype_snake import Snake


class LearningQ:
    """
    Prototype for the q-learning algorithm
    Author: Adam Ross
    Date: 01/05/2019
    """

    ETA = 0.1  # learning rate
    GAMMA = 0.95  # discount
    EPSILON = 1  # exploration: greedy - random
    REWARD = 0  # default reward
    EPISODES = 200  # number of episodes
    SPEED = 600  # the speed the snake moves
    SIZE = 1  # the length of the snake
    DIRS = {0: '\x1b[A', 1: '\x1b[B', 2: '\x1b[C', 3: '\x1b[D'}
    DISPLAY = False

    def __init__(self):
        """
        Class Initializer
        """
        self.game = Snake(True)  # snake game instance
        self.ql_map = [[[0, 0, 0, 0] for _ in range(self.game.N)]
                       for _ in range(self.game.N)]  # matrix of states
        self.agent = None  # the agent
        self.trophy = None  # the trophy with reward = 1
        self.state = []  # the state the action q-value is being updated
        self.nxt_state = []  # the adjacent state action is directed to
        self.action = None  # the agent action; direction from the state
        self.reward = self.REWARD

    def update_state(self):
        """
        Updates a state action q-value
        """
        if self.state == [self.agent[0][0], self.agent[0][1]]:
            nxt_state = [0]
        else:
            if self.game.snake_crashed:
                self.reward, nxt_state = -1, [0]
            elif [self.agent[0][0], self.agent[0][1]] == self.trophy:
                self.reward, nxt_state = 1, [0]
            else:
                nxt_state = self.ql_map[self.agent[0][0]][self.agent[0][1]].copy()
        self.ql_map[self.state[0]][self.state[1]][self.action] += \
            self.ETA * (self.reward + self.GAMMA * max(nxt_state) -
                        self.ql_map[self.state[0]][self.state[1]][self.action])

    def play_episode(self):
        """
        Plays an episode until agent finds trophy or
        """
        self.state = [self.agent[0][0], self.agent[0][1]]
        self.action = randint(0, 3)
        self.game.move_snake(self.DIRS[self.action])
        self.update_state()

        if self.DISPLAY:
            self.game.display_snake()

        if not self.game.snake_crashed and not \
                [self.agent[0][0], self.agent[0][1]] == self.trophy:
            sleep(1 / self.SPEED)
            self.play_episode()

    def q_learn(self):
        """
        Runs the Q-learning algorithm for number of given episodes
        :return:
        """
        for _ in range(self.EPISODES):
            self.game = Snake(True)
            self.agent = self.game.snake
            self.trophy = self.game.food

            if len(self.agent) > self.SIZE:
                self.agent.pop()

            while len(self.agent) < self.SIZE:
                self.game.add_tail()

                for i in range(2):
                    if not 0 <= self.agent[-1][0] < self.game.N:
                        self.agent.pop()
                        break
            self.play_episode()


if __name__ == "__main__":
    app = LearningQ()
    app.q_learn()
    [print(row) for row in app.ql_map]
