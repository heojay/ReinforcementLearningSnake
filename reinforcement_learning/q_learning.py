#!/usr/bin/env python3

from reinforcement_learning.reinforcement_learning import ReinforcementLearning


class QLearning(ReinforcementLearning):
    """
    Class for Q-learning restricted learning algorithm
    Author: Adam Ross
    Date: 05/05/2019
    """

    def __init__(self, eta, gamma, epsilon):
        """
        Class initializer
        """
        super().__init__(eta, gamma, epsilon)

    def update_state(self):
        """
        Updates a state action q-value using the Q-learning update rule
        """
        if self.state != self.agent[0]:
            if self.game.snake_crashed:
                self.reward, nxt_state = self.NEGATIVE_REWARD, [0]
            elif self.agent[0] == self.trophy:
                self.reward, nxt_state = self.POSITIVE_REWARD, [0]
            else:
                nxt_state = self.rl_map[self.agent[0][0]][self.agent[0][1]].\
                    copy()
            state = self.rl_map[self.state[0]][self.state[1]].copy()
            self.rl_map[self.state[0]][self.state[1]][self.action] += \
                self.eta * (self.reward + self.gamma * max(nxt_state) -
                            state[self.action])
            self.reward = self.DEFAULT_REWARD  # reset reward to default
