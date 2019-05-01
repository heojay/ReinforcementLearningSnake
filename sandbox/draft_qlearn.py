# qlearn.py
# Implementation of q-learning
# 2019-04-22

import numpy as np

class Qlearn:
    lr = 0.1 # alpha learn_rate
    disc = 0.95  # gamma discunt
    expl = 1.0 # exploration
    reward = 0 # default_reward

    def __init__(self, width, height):
        self.w = width
        self.h = height        
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


def main():
    global lr, disc, expl, reward

    print("Begin qlearn")
    width = 10
    height = 10
    q = Qlearn(width, height)
    print(q.mtx_north)
    v = q.getQValueEast(1,1)
    print(v)
    cellmax = q.getMaxValue(1,1)
    print(cellmax.maxdir)
    print(cellmax.maxval)

    newqval = q.getUpdateQValue(1, 1, 0.5)
    print(newqval)

    print("End qlearn")


main()
