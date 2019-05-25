#!/usr/bin/env python3

# Trains the snake game using Q-learning
# The Q-learning hyperparameters are set in snake_qlearn.py

import sys
import time
import numpy as np
import snake_qlearn as env
from sys import argv as arguments


class SnakeTrainer:
    """
    Class for 
    Date: 2019-05-21
    """

    DISPLAY = False # whether to display the game during training (last 10%)
    PATH = 'C:/Dev/logs/'  # ex: C:/Dev/logs/
    Q_LEARNING_FILE = 'q-learning.txt'

    def __init__(self):
        """
        Class initializer
        """
        self.qlearn = env.SnakeQlearning(False, self.DISPLAY)
        self.n_episodes = -1

    def train(self, save_qfile=True, save_log=False):
        """
        Trains the snake game using Q-learning
        :param save_qfile: whether to save a qdata file
        :param save_log: whether to save a rolling log file
        """
        if save_qfile:
            qfile = str.format("{0}{1}", self.PATH, self.Q_LEARNING_FILE)
        else:
            qfile = None

        self.n_episodes = self.qlearn.train(qfile)

        if save_log:
            file_name = str.format("rl_gymsnake_{0}.log", time.strftime('%Y%m%d_%H%M%S'))
            self.qlearn.save_log(str.format("{0}{1}", self.PATH, file_name))

    def display_training_result(self):
        """
        """
        #app.display_qvalues()

        if not self.qlearn.q.check_all_states_nonzero():
            print("Training did not succeed. All states are not non zero.")

        if self.qlearn.q.check_all_states_nonzero():
            print("Now verifying the training using all states")    
            try:
                invalid_states = self.qlearn.q.verify_all_states(self.qlearn.gl_metrics['trophy'])
                print("The number of states with invalid paths:{0}. {1}".format(len(invalid_states), invalid_states))
            except Exception as e:
                print("Unable to determine best path from snake to trophy.")
                print(e)

            print("Finished game level after {0} training episodes".format(self.n_episodes))

        self.qlearn.plot_training_scores()

    def replay(self):
        """
        Plays back the optimal paths from previous training
        """
        start_coord = None
        frame_speed = 0.3
        self.qlearn.replay(str.format("{0}{1}", self.PATH, self.Q_LEARNING_FILE), start_coord, frame_speed)



if __name__ == "__main__":
    start = time.time()

    app = SnakeTrainer()
    
    if "-t" in arguments:
        # train / learn
        app.train()
        end = time.time()
        app.display_training_result()
        print("Total train duration:{0} seconds" .format(round(end - start, 3)))
    elif "-r" in arguments:
        app.replay()
    elif "-m" in arguments:
        # manual human play
        print("TODO: manual play")
    else:
        print("Enter an argument: -t or -r or -m")


