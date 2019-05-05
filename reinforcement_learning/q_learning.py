#!/usr/bin/env python3

from reinforcement_learning.reinforcement_learning import ReinforcementLearning


class QLearning(ReinforcementLearning):
    """
    Class for Q-learning restricted learning algorithm
    Author: Adam Ross
    Date: 05/05/2019
    """

    def __init__(self, file_name):
        """
        Class initializer
        :param file_name: the name of the file the learning data is saved to
        """
        super().__init__(file_name)

    def update_state(self):
        """
        Updates a state action q-value
        """
        self.state = self.agent[0].copy()
        self.choose_action()
        self.game.move_snake(self.game.absolute_dirs(self.DIRS[self.action]))

        if self.game.snake_crashed:
            self.reward, nxt_state = self.NEGATIVE_REWARD, [0]
        elif self.agent[0] == self.trophy:
            self.reward, nxt_state = self.POSITIVE_REWARD, [0]
        else:
            nxt_state = self.rl_map[self.agent[0][0]][self.agent[0][1]].\
                copy()
        self.update_rule(self.state, self.action, max(nxt_state))
        self.reward = self.DEFAULT_REWARD  # reset reward to default
