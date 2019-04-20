#!/usr/bin/env python3

from random import randint
from time import sleep
from sys import exit, argv
from os import system


class Snake:
    """
    Prototype for the snake game
    @author: Adam Ross
    @date: 20/04/2019
    """

    N = 10  # the size of the snake game environment
    TAIL_CHANCE = 5  # size of odds snake has chance of growing extra 'blocks'
    SLEEP_TIME = 2  # seconds that the game is paused between snake moves

    def __init__(self, manual=False):
        """
        Initialize the Snake class
        :param manual: Boolean for True if manual play, False if automated
        """
        self.snake = [[randint(self.N // 3, self.N - self.N // 3),
                       randint(self.N // 3, self.N - self.N // 3)]]
        self.add_tail()
        self.direction = ""
        self.snake_grew = False
        self.manual = manual

    def move_snake(self, dir):
        """
        Moves snake one space forward, left or right of current head position
        :param dir: the direction the snake is moving; forward, left or right
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

        if self.not_crashed(head):
            self.snake.pop()
            self.snake[:0] = [head]
            return True
        return False

    def add_tail(self):
        """
        Adds an additional 'block' to the tail of the snake
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
        else:
            tail = [self.snake[-1][0] + (self.snake[-1][0] - self.snake[-2][0]),
                    self.snake[-1][1] + (self.snake[-1][1] - self.snake[-2][1])]

        if self.not_crashed(tail):
            self.snake.append(tail)
            self.snake_grew = True

    def not_crashed(self, b):
        """
        Determines if a snakes move/tail expansion does not result in a crash
        :param b: new 'block' being added to snake at either head or tail end
        :return: True if 'block' pos is available inside game, False otherwise
        """
        return 0 <= b[0] < self.N and 0 <= b[1] < self.N and b not in self.snake

    def display_snake(self):
        """
        Displays the snake in the snake game environment
        """
        system('clear')
        [print([' ' if [j, i] not in self.snake else '#' for i in
                range(self.N)]) for j in range(self.N)]

        if self.snake_grew:
            print("Snake grew an extra block to its tail")
            self.snake_grew = False
        print(self.direction)

    def go(self):
        """
        Plays the snake game manually/automated until the snake crashes
        """
        self.display_snake()  # displays the snake and the board

        if not randint(0, self.TAIL_CHANCE):
            self.add_tail()

        if self.manual:
            move = input("Enter 1 for forward, 2 for left, 3 for right:\n")

            while move not in ['1', '2', '3']:
                move = input("Enter 1 for forward, 2 for left, 3 for right:\n")
        else:
            sleep(self.SLEEP_TIME)  # pauses game between snake moves
            move = randint(1, 3)

        if self.move_snake(int(move)):  # if snake move without crashing
            self.go()

    def stop(self):
        """
        Displays final game state, then exits program
        """
        self.display_snake()  # displays the snake and the board
        print("Snake crashed")
        exit(1)


if __name__ == "__main__":
    if len(argv) > 1 and argv[1] == "-m":
        app = Snake(True)
    else:
        app = Snake()
    app.go()
    app.stop()
