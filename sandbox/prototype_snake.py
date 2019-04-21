#!/usr/bin/env python3

from random import randint
from time import sleep
from sys import exit, argv
from os import system


class Snake:
    """
    Prototype for the snake game
    @author: Adam Ross
    @date: 22/04/2019
    """

    N = 10  # the size of the snake game environment
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
        self.snake = [[randint(self.N // 3, self.N - self.N // 3),
                       randint(self.N // 3, self.N - self.N // 3)]]
        self.add_tail()
        self.direction = ""
        self.snake_grew = False
        self.snake_ate = False
        self.snake_crashed = False
        self.food = self.new_food()
        self.manual = manual

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

    def new_food(self):
        """
        Sets a new position for a piece of food for the snake to eat
        :return: the new food position
        """
        new_food_pos = [randint(0, self.N - 1), randint(0, self.N - 1)]

        while new_food_pos in self.snake:
            new_food_pos = [randint(0, self.N - 1), randint(0, self.N - 1)]
        return new_food_pos

    def move_snake(self, dir):
        """
        Moves snake one space forward, left or right of current head position
        :param dir: the direction the snake is moving; forward, left or right
        """
        if dir == 1:  # snake is continuing straight
            head = self.move_forward()
        elif dir == 2:  # snake is turning left
            head = self.move_left()
        else:  # snake is turning right
            head = self.move_right()
        tail = self.snake.pop()

        if self.not_crashed(head):
            self.snake[:0] = [head]
        else:
            self.snake.append(tail)
            self.snake_crashed = True

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
        else:
            self.snake_crashed = True
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
        if self.snake[0] == self.food:
            self.add_tail()
            self.food = self.new_food()
            self.snake_ate = True

        if self.snake_crashed:
            game = [[' ' if [j, i] not in self.snake else self.RED
                     for i in range(self.N)] for j in range(self.N)]
        else:
            game = [[' ' if [j, i] not in self.snake else self.YELLOW
                     for i in range(self.N)] for j in range(self.N)]
            game[self.food[0]][self.food[1]] = self.FOOD
            game[self.snake[0][0]][self.snake[0][1]] = self.PURPLE
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
            move = input("Enter 1 for forward, 2 for left, 3 for right:\n")

            while move not in ['1', '2', '3']:
                move = input("Enter 1 for forward, 2 for left, 3 for right:\n")
            self.move_snake(int(move))
        elif not self.snake_crashed:
            sleep(self.SLEEP_TIME)  # pauses game between snake moves
            self.move_snake(randint(1, 3))

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


if __name__ == "__main__":
    if len(argv) > 1 and argv[1] == "-m":
        app = Snake(True)
    else:
        app = Snake()
    app.go()
