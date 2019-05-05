#!/usr/bin/env python3

from reinforcement_learning.snake.prototype_snake import Snake
from reinforcement_learning.q_learning import QLearning
from reinforcement_learning.sarsa import Sarsa
from sys import argv as arguments


class ReinforcementLearningAlgorithms:
    """
    Class for reinforcement learning algorithms in snake game environment
    Author: Adam Ross
    Date: 05/05/2019
    """

    SIZE = 3  # the initial size of the snake in the snake game
    Q_LEARNING_FILE = 'q-learning.txt'
    SARSA_FILE = 'sarsa.txt'

    def __init__(self):
        """
        Class initializer
        """
        self.q_learning = QLearning(self.Q_LEARNING_FILE)
        self.sarsa = Sarsa(self.SARSA_FILE)
        self.snake = Snake(True)

    def learn_q_learning(self):
        """
        Runs the q-learning algorithm in the snake game environment
        """
        self.q_learning.learn()

    def test_q_learning(self):
        """
        Tests the q-learning algorithm output data by drawing optimal paths
        """
        self.q_learning.optimal_paths()

    def learn_sarsa(self):
        """
        Runs the SARSA algorithm in the snake game environment
        """
        self.sarsa.learn()

    def test_sarsa(self):
        """
        Tests the SARSA algorithm output data by drawing optimal paths
        """
        self.sarsa.optimal_paths()

    def play_snake(self):
        """
        Plays a manual snake game
        """
        self.snake.set_food([2, 3])
        self.snake.set_snake()

        for i in range(app.SIZE):
            self.snake.add_tail(True)
        self.snake.go()


if __name__ == "__main__":
    app = ReinforcementLearningAlgorithms()

    if "-q" in arguments:
        if "-l" in arguments:
            app.learn_q_learning()
        elif "-o" in arguments:
            app.test_q_learning()
    elif "-s" in arguments:
        if "-l" in arguments:
            app.learn_sarsa()
        elif "-o" in arguments:
            app.test_sarsa()
    elif "-m" in arguments:
        app.play_snake()
