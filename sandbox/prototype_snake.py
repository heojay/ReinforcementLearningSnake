#!/usr/bin/env python3

from random import randint
from time import sleep
from sys import exit
from os import system

N = 10  # the size of the snake game environment


class Snake:
    """
    Prototype for the snake game
    @author: Adam Ross
    @date: 18/04/2019
    """

    TAIL_CHANCE = 5  # size of odds snake has chance of growing extra 'blocks'
    SLEEP_TIME = 2  # seconds that the game is paused between snake moves

    def __init__(self, n=10):
        """
        Initialize the Snake class
        :param n: the size of the snake game environment
        """
        self.snake = [[randint(n // 3, n - n // 3), randint(n // 3, n - n // 3)]]
        self.add_tail(n)
        self.direction = ""
        self.go(n)

    def move_snake(self, dir, n):
        """
        Moves snake one space forward, left or right of current head position
        :param dir: the direction the snake is moving; forward, left or right
        :param n: the size of the snake game environment
        :return: True if the snake moves without crashing, False otherwise
        """
        if dir == 1:  # snake is continuing straight
            head = [self.snake[0][0] + (self.snake[0][0] - self.snake[1][0]),
                    self.snake[0][1] + (self.snake[0][1] - self.snake[1][1])]
            self.direction = "Snake continued straight"
        elif dir == 2:  # snake is turning left
            head = [self.snake[0][0] - (self.snake[0][1] - self.snake[1][1]),
                    self.snake[0][1] + (self.snake[0][0] - self.snake[1][0])]
            self.direction = "Snake turned left"
        else:  # snake is turning right
            head = [self.snake[0][0] + (self.snake[0][1] - self.snake[1][1]),
                    self.snake[0][1] - (self.snake[0][0] - self.snake[1][0])]
            self.direction = "Snake turned right"
        self.snake.pop()

        if self.not_crashed(head, n):
            self.snake[:0] = [head]
            return True
        return False

    def add_tail(self, n):
        """
        Adds an additional 'block' to the tail of the snake
        :param n: the size of the snake game environment
        """
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
        """
        Determines if a snakes move/tail expansion does not result in a crash
        :param blc: new 'block' being added to snake at either head or tail end
        :param n: the size of the snake game environment
        :return: True if 'block' pos is available inside game, False otherwise
        """
        return 0 <= blc[0] < n and 0 <= blc[1] < n and blc not in self.snake

    def display_snake(self, n):
        """
        Displays the snake in the snake game environment
        :param n: the size of the snake game environment
        """
        system('clear')
        [print([' ' if [j, i] not in self.snake else '#' for i in range(n)])
         for j in range(n)]
        print(self.direction)

    def go(self, n):
        """
        Recursively plays the snake game until the snake crashes
        :param n: the size of the snake game environment
        """
        self.display_snake(n)  # displays the snake and the board

        if not randint(0, self.TAIL_CHANCE):
            self.add_tail(n)
        sleep(self.SLEEP_TIME)  # pauses game between snake moves

        if self.move_snake(randint(1, 3), n):  # if snake move without crashing
            self.go(n)
        else:
            self.display_snake(n)  # displays the snake and the board
            print("Snake crashed")
            exit(1)


if __name__ == "__main__":
    Snake(N)
