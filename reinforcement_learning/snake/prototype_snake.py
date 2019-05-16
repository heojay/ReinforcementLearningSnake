#!/usr/bin/env python3

from random import randint
from time import sleep
from sys import exit
from os import system


class Snake:
    """
    Simple snake game for reinforcement learning
    @author: Adam Ross
    @date: 02/05/2019
    """

    N = 11  # the size of the snake game environment
    SLEEP_TIME = 0.5  # seconds that the game is paused between snake moves
    YELLOW = '\033[93m' + '#' + '\033[0m'
    PURPLE = '\033[95m' + '#' + '\033[0m'
    RED = '\033[31m' + '#' + '\033[0m'
    FOOD = '\033[92m' + '@' + '\033[0m'

    def __init__(self, manual=False):
        """
        Initialize the Snake class
        :param manual: Boolean for True if manual play, False if automated
        """
        self.food = None
        self.snake = None
        self.tail = None
        self.direction = ""
        self.snake_grew = False
        self.snake_ate = False
        self.snake_crashed = False
        self.manual = manual
        self.restricted_learning = False

    def set_food(self, new_pos):
        """
        Initializes the food position
        :param new_pos: the [x, y] of the new food position
        """
        self.food = new_pos

    def set_snake(self):
        """
        Randomly initializes snake position on board not equal to food position
        """
        self.snake = [[randint(0, self.N - 1), randint(0, self.N - 1)]]

        while self.food in self.snake:
            self.snake = [[randint(0, self.N - 1), randint(0, self.N - 1)]]

    def move_forward(self):
        """
        Calculates the next position of snake head when moving forward
        :return: the next position of the snakes head
        """
        self.direction = "Snake continued straight"
        return [self.snake[0][0] + (self.snake[0][0] - self.snake[1][0]),
                self.snake[0][1] + (self.snake[0][1] - self.snake[1][1])]

    def move_left(self):
        """
        Calculates the next position of snake head when moving left
        :return: the next position of the snakes head
        """
        self.direction = "Snake turned left"
        return [self.snake[0][0] - (self.snake[0][1] - self.snake[1][1]),
                self.snake[0][1] + (self.snake[0][0] - self.snake[1][0])]

    def move_right(self):
        """
        Calculates the next position of snake head when moving right
        :return: the next position of the snakes head
        """
        self.direction = "Snake turned right"
        return [self.snake[0][0] + (self.snake[0][1] - self.snake[1][1]),
                self.snake[0][1] - (self.snake[0][0] - self.snake[1][0])]

    def move_north(self):
        """
        Calculates the next position of snake head when moving North
        :return: the next position of the snakes head
        """
        if len(self.snake) == 1 or [self.snake[0][0] - 1, self.snake[0][1]] !=\
                self.snake[1]:
            self.direction = "Snake moved North"
            return [self.snake[0][0] - 1, self.snake[0][1]]
        return None

    def move_south(self):
        """
        Calculates the next position of snake head when moving South
        :return: the next position of the snakes head
        """
        if len(self.snake) == 1 or [self.snake[0][0] + 1, self.snake[0][1]] !=\
                self.snake[1]:
            self.direction = "Snake moved South"
            return [self.snake[0][0] + 1, self.snake[0][1]]
        return None

    def move_east(self):
        """
        Calculates the next position of snake head when moving East
        :return: the next position of the snakes head
        """
        if len(self.snake) == 1 or [self.snake[0][0], self.snake[0][1] + 1] !=\
                self.snake[1]:
            self.direction = "Snake moved East"
            return [self.snake[0][0], self.snake[0][1] + 1]
        return None

    def move_west(self):
        """
        Calculates the next position of snake head when moving West
        :return: the next position of the snakes head
        """
        if len(self.snake) == 1 or [self.snake[0][0], self.snake[0][1] - 1] !=\
                self.snake[1]:
            self.direction = "Snake moved West"
            return [self.snake[0][0], self.snake[0][1] - 1]
        return None

    def new_food(self):
        """
        Sets a new position for a piece of food for the snake to eat
        :return: the new food position
        """
        food = [self.food[1], (self.food[0] + self.food[1]) % self.N]

        while food in self.snake:
            food = [food[1], (food[0] + food[1]) % self.N]
        return food

    def relative_dirs(self, dir):
        """
        Moves snake one space forward, left or right of current head position
        :param dir: the new snake head position
        """
        if str(dir) == '1':  # snake is continuing straight
            return self.move_forward()
        elif str(dir) == '2':  # snake is turning left
            return self.move_left()
        else:  # snake is turning right
            return self.move_right()

    def absolute_dirs(self, dir):
        """
        Moves snake in absolute N, E, S, or W direction of current head pos
        :param dir: the direction the snake is moving
        :return: the new snake head position or None if not able to move
        """
        if dir == '\x1b[A':  # snake is moving North
            return self.move_north()
        elif dir == '\x1b[B':  # snake is moving South
            return self.move_south()
        elif dir == '\x1b[C':  # snake is moving East
            return self.move_east()
        else:  # snake is moving West
            return self.move_west()

    def move_snake(self, head_move):
        """
        Determines if snake can move and moves snake if so
        :param head_move: if and where the snake is moving
        """
        if head_move:
            self.tail = self.snake.pop()

            if self.not_crashed(head_move):
                self.snake[:0] = [head_move]
            else:
                self.snake.append(self.tail)
                self.snake_crashed = True
        else:
            self.direction = "Snake can't move inside itself"

    def add_tail(self, start=False):
        """
        Adds an additional 'block' to the tail of the snake
        :param start: is the start of the game initializing of snake
        """
        if start:
            while True:
                if randint(0, 1):
                    if randint(0, 1):
                        tail = [self.snake[-1][0], self.snake[-1][1] + 1]
                    else:
                        tail = [self.snake[-1][0], self.snake[-1][1] - 1]
                else:
                    if randint(0, 1):
                        tail = [self.snake[-1][0] + 1, self.snake[-1][1]]
                    else:
                        tail = [self.snake[-1][0] - 1, self.snake[-1][1]]
                if 0 <= tail[0] < self.N and 0 <= tail[1] < self.N:
                    break
        else:
            tail = self.tail

        if self.not_crashed(tail):
            self.snake.append(tail)
        else:
            self.snake_crashed = True
        self.snake_grew = True

    def not_crashed(self, b):
        """
        Determines if a snakes move/tail expansion does not result in a crash
        :param b: new 'block' being added to snake at either head or tail end
        :return: True if 'block' pos is available inside game, False otherwise
        """
        return 0 <= b[0] < self.N and 0 <= b[1] < self.N and \
                    b not in self.snake[1:]

    def display_snake(self):
        """
        Displays the snake in the snake game environment
        """
        if self.snake[0] == self.food:
            self.add_tail()
            self.set_food(self.new_food())
            self.snake_ate = True

        if self.snake_crashed:
            game = [[' ' if [j, i] not in self.snake else self.RED
                     for i in range(self.N)] for j in range(self.N)]
        else:
            game = [[' ' if [j, i] not in self.snake else self.YELLOW
                     for i in range(self.N)] for j in range(self.N)]
            game[self.food[0]][self.food[1]] = self.FOOD
            game[self.snake[0][0]][self.snake[0][1]] = self.PURPLE

        if not self.restricted_learning:
            system('clear')
        print('-' * (self.N + 2))
        [print('|' + "".join([j for j in i]) + '|') for i in game]
        print('-' * (self.N + 2))
        print(self.direction)

        if self.snake_ate:
            print("Snake ate an apple")
            self.snake_ate = False

        if self.snake_grew:
            print("Snake grew an extra block to its tail")
            self.snake_grew = False

    def go(self):
        """
        Plays the snake game manually/automated until the snake crashes
        """
        self.display_snake()  # displays the snake and the board

        if self.manual and not self.snake_crashed:
            move = input("Use the arrow keys to move the snake:\n")

            while move not in ['\x1b[A', '\x1b[B', '\x1b[C', '\x1b[D']:
                move = input("Use the arrow keys to move the snake:\n")
            self.move_snake(self.absolute_dirs(move))
        elif not self.snake_crashed:
            sleep(self.SLEEP_TIME)  # pauses game between snake moves
            self.move_snake(self.relative_dirs(randint(1, 3)))

        if not self.snake_crashed:  # if snake move without crashing
            self.go()
        else:
            self.stop()

    def stop(self):
        """
        Displays final game state, then exits program
        """
        self.display_snake()  # displays the snake and the board
        print("Snake crashed")
        exit(1)
