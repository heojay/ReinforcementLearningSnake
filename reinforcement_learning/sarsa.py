#!/usr/bin/env python3

from reinforcement_learning.reinforcement_learning import ReinforcementLearning


class Sarsa(ReinforcementLearning):
    """
    Class for SARSA restricted learning algorithm
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
        if self.action not in [0, 1, 2, 3]:
            self.state = self.agent[0].copy()
            self.choose_action()
        self.game.move_snake(self.game.absolute_dirs(self.DIRS[self.action]))
        cur_action = self.action

        if self.game.snake_crashed:
            self.reward, nxt_action = self.NEGATIVE_REWARD, 0
        elif self.agent[0] == self.trophy:
            self.reward, nxt_action = self.POSITIVE_REWARD, 0
        else:
            self.choose_action()
            nxt_action = self.rl_map[self.agent[0][0]][self.
                                                       agent[0][1]][self.
                                                                    action]

        if self.state != self.trophy:
            self.update_rule(self.state, cur_action, nxt_action)
        self.reward, self.state = self.DEFAULT_REWARD, self.agent[0].copy()
