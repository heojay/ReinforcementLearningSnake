# qlearn.py
# Implementation of q-learning
# 2019-04-22

class qlearn:
    learn_rate = 0.1 # alpha
    discount = 0.95  # gamma
    exploration = 1.0
    default_reward = 0

    def __init__(self, width, height):
        self.w = width
        self.h = height
        ###Matrix [[0 for x in range(self.w)] for y in range(self.h)] 
        mtx = [[1, 2, 3, 4], [7, 8, 9]]
        self.m = mtx

    def displayMatrix(self):    
        print(self.m[0][0])


def main():
    print("Begin qlearn")
    width = 10
    height = 10
    q = qlearn(width, height)
    q.displayMatrix()
    print("End qlearn")


main()
