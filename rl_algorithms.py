#!/usr/bin/env python3

from reinforcement_learning.snake.prototype_snake import Snake
from reinforcement_learning.q_learning import QLearning
from reinforcement_learning.sarsa import Sarsa
from matplotlib import patches, pyplot as plt
from sys import argv as arguments


class ReinforcementLearningAlgorithms:
    """
    Class for reinforcement learning algorithms in snake game environment
    Author: Adam Ross
    Date: 22/05/2019
    """

    SIZE = 5  # the initial size of the snake/agent whether playing or learning
    LEVELS = 1  # the number of snake game levels being learned
    EPISODES = 15000  # the number of learning episodes per snake game level
    Q_LEARNING_FILE = 'q-learning_L5.txt'
    SARSA_FILE = 'sarsa_L5.txt'

    def __init__(self):
        """
        Class initializer
        """
        self.cols, self.algos = ['red', 'blue'], ['Q-learning', 'SARSA']
        self.q_learning = QLearning(self.Q_LEARNING_FILE, self.EPISODES,
                                    self.LEVELS, self.SIZE)
        self.sarsa = Sarsa(self.SARSA_FILE, self.EPISODES, self.LEVELS,
                           self.SIZE)
        self.snake = Snake(True)

    def learn_q_learning(self):
        """
        Runs the q-learning algorithm in the snake game environment
        """
        return self.q_learning.learn()

    def test_q_learning(self):
        """
        Tests the q-learning algorithm output data by drawing optimal paths
        """
        return self.q_learning.optimal_paths()

    def learn_sarsa(self):
        """
        Runs the SARSA algorithm in the snake game environment
        """
        return self.sarsa.learn()

    def test_sarsa(self):
        """
        Tests the SARSA algorithm output data by drawing optimal paths
        """
        return self.sarsa.optimal_paths()

    def play_snake(self):
        """
        Plays a manual snake game
        """
        self.snake.set_food([2, 3])
        self.snake.set_snake()

        for i in range(self.SIZE):
            self.snake.add_tail(True)
        self.snake.go()

    def plot_learning(self, q_learn, sarsa):
        """
        Plots a line graph of the number of trophies found while learning
        :param q_learn: the q-learning learning data
        :param sarsa: the SARSA learning data
        """
        if self.LEVELS == 1:
            plt.plot([j for j in range(0, self.EPISODES,
                                       int(self.EPISODES / 10))], q_learn,
                     color=self.cols[0])
            plt.plot([j for j in range(0, self.EPISODES,
                                       int(self.EPISODES / 10))], sarsa,
                     color=self.cols[1])
            plt.ylabel("Trophy count in every " +
                       str(int(self.EPISODES / 10)) + " episodes")
            plt.title("Count of trophies found while learning " +
                      str(self.EPISODES) + " episodes at level " +
                      str(self.SIZE))
        else:
            plt.plot([j for j in range(0, int(self.EPISODES * self.LEVELS),
                                       self.EPISODES)], q_learn,
                     color=self.cols[0])
            plt.plot([j for j in range(0,  int(self.EPISODES * self.LEVELS),
                                       self.EPISODES)], sarsa,
                     color=self.cols[1])
            plt.ylabel("Trophy count in every " +
                       str(int(self.EPISODES / self.LEVELS)) + " episodes")
            plt.title("Count of trophies found while learning " +
                      str(self.LEVELS) + " snake levels")
        plt.legend(handles=[patches.Patch(color=self.cols[i],
                                          label=self.algos[i])
                            for i in range(2)])
        plt.xlabel("Episodes")
        plt.show()

    def plot_testing(self, q_learn, sarsa):
        """
        Plots the number of moves to trophy from each state during playback
        :param q_learn: the q-learning learning data
        :param sarsa: the SARSA learning data
        """
        if self.LEVELS > 1:
            plt.title("Agent moves to trophy in " + str(self.LEVELS) +
                      " levels of " + str(self.EPISODES) + " episodes")
            plt.ylabel("Moves to trophy per " + str(self.EPISODES) +
                       " episode level")
        else:
            plt.title("Agent moves to trophy in " + str(self.EPISODES) +
                      " episodes at level " + str(self.SIZE))
            plt.ylabel("Moves to trophy per " + str(self.EPISODES) +
                       " episodes")
        plt.xlabel("Agent starting cell: from pos(0, 0) to pos(10, 10)")
        plt.plot([j for j in range(121)], q_learn, color=self.cols[0])
        plt.plot([j for j in range(121)], sarsa, color=self.cols[1])
        plt.legend(handles=[patches.Patch(color=self.cols[i],
                                          label=self.algos[i])
                            for i in range(2)])
        plt.show()


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
    else:
        sarsa = app.learn_sarsa()
        sarsa_test = app.test_sarsa()
        q_learn = app.learn_q_learning()
        q_learn_test = app.test_q_learning()
        app.plot_learning(q_learn, sarsa)
        app.plot_testing(q_learn_test, sarsa_test)
