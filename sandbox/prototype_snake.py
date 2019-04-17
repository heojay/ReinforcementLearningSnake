#!/usr/bin/env python3

from random import randint
from time import sleep
from sys import exit
from os import system


class Snake:

    TAIL_CHANCE = 5  # chance of snake growing an extra block to the tail

    def __init__(self, n=10):
        self.snake = [[randint(2, n - 3), randint(2, n - 3)]]
        self.add_tail(n)
        self.go(n)

    def move_snake(self, dir, n):
        if dir == 1:  # snake is continuing straight
            head = [self.snake[0][0] + (self.snake[0][0] - self.snake[1][0]),
                    self.snake[0][1] + (self.snake[0][1] - self.snake[1][1])]
        elif dir == 2:  # snake is turning left
            head = [self.snake[0][0] - (self.snake[0][1] - self.snake[1][1]),
                    self.snake[0][1] + (self.snake[0][0] - self.snake[1][0])]
        else:  # snake is turning right
            head = [self.snake[0][0] + (self.snake[0][1] - self.snake[1][1]),
                    self.snake[0][1] - (self.snake[0][0] - self.snake[1][0])]
        self.snake.pop()

        if self.not_crashed(head, n):
            self.snake[:0] = [head]
            return True
        return False

    def add_tail(self, n):
        if len(self.snake) == 1:
            if randint(0, 1):
                if randint(0, 1):
                    tail = [self.snake[0][0], self.snake[0][1] + 1]
                else:
                    tail = [self.snake[0][0], self.snake[0][1] - 1]
            else:
                if randint(0, 1):
                    tail = [self.snake[0][0] + 1, self.snake[0][1]]
                else:
                    tail = [self.snake[0][0] + 1, self.snake[0][1]]
            self.snake.append(tail)
        tail = [self.snake[-1][0] + (self.snake[-1][0] - self.snake[-2][0]),
                self.snake[-1][1] + (self.snake[-1][1] - self.snake[-2][1])]

        if self.not_crashed(tail, n):
            self.snake.append(tail)

    def not_crashed(self, blc, n):
        return 0 <= blc[0] <= n and 0 <= blc[1] <= n and blc not in self.snake

    def display_snake(self, n):
        system('clear')
        [print([' ' if [j, i] not in self.snake else '#' for i in range(n)])
         for j in range(n)]

    def go(self, n):
        self.display_snake(n)  # displays the snake and the board

        if not randint(0, self.TAIL_CHANCE):
            self.add_tail(n)
        sleep(5)  # pause 5 seconds before snake moves

        if self.move_snake(randint(1, 3), n):  # if snake move without crashing
            self.go(n)
        else:
            print("Snake crashed")
            exit(1)


if __name__ == "__main__":
    Snake()
