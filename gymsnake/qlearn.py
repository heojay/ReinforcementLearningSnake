# qlearn.py
# Implementation of q-learning
# 2019-05-01

import numpy as np

class Qlearn:
    lr = 0.1 # alpha learn_rate
    disc = 0.95  # gamma discunt
    expl = 1.0 # exploration
    reward = 0 # default_reward

    def __init__(self, width, height, lr, disc, expl, reward):
        self.w = width
        self.h = height
        self.lr = lr
        self.disc = disc
        self.expl = expl
        self.reward = reward
        self.mtx_north = np.random.rand(height, width)
        self.mtx_south = np.random.rand(height, width)
        self.mtx_east = np.random.rand(height, width)
        self.mtx_west = np.random.rand(height, width)

    def getUpdateQValue(self, row, col, oldval):
        cellmax = self.getMaxValue(row, col)
        learnedval = (self.reward + self.disc * cellmax.maxval )
        qv = (1-self.lr) * oldval + self.lr * learnedval
        return qv

    def getMaxValue(self, row, col):
        v = self.mtx_north[row, col]
        qc = Qcell(row, col, v, "north")
        if self.mtx_south[row, col] > v:
            v = self.mtx_south[row, col]
            qc = Qcell(row, col, v, "south")
        if self.mtx_east[row, col] > v:
            v = self.mtx_east[row, col]
            qc = Qcell(row, col, v, "east")
        if self.mtx_west[row, col] > v:
            v = self.mtx_west[row, col]
            qc = Qcell(row, col, v, "west")
        return qc

    def getQValueNorth(self, row, col):
        return self.mtx_north[row, col]

    def getQValueSouth(self, row, col):
        return self.mtx_south[row, col]

    def getQValueEast(self, row, col):
        return self.mtx_east[row, col]

    def getQValueWest(self, row, col):
        return self.mtx_west[row, col]

    ##def displayMatrix(self, ):    
        ###print(self.m[0][0])


class Qcell:
    def __init__(self, row, col, maxval, maxdir):
        self.row = row
        self.col = col
        self.maxval = maxval
        self.maxdir = maxdir



"""
Example usage:
    q = Qlearn(width, height)
    v = q.getQValueEast(1,1)
    cellmax = q.getMaxValue(1,1)
    newqval = q.getUpdateQValue(1, 1, 0.5)
"""