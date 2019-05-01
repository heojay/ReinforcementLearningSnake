# Uses the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a blind version Q-learning version
# Note that this only works for one food goal

# TODO: look into GoalEnv

import gym
import gym_snake
from gymsnake import Qlearn
#import gymsnake.qlearn


class SnakeQlearning:

    ETA = 0.1  # learning rate
    GAMMA = 0.95  # discount
    EPSILON = 1  # exploration: greedy - random
    REWARD = 0  # default reward
    EPISODES = 200  # number of episodes

    def __init__(self):
        """
        Class Initializer
        """
        self.env = gym.make('snake-v0')
        self.env.n_foods = 1
        self.env.random_init = False
        width = self.env.grid_size[0]
        height = self.env.grid_size[1]
        self.qlearn = Qlearn(width, height, self.ETA, self.GAMMA, self.EPSILON, self.REWARD)

        ##env.reset()
        ##game_controller = env.controller
        ##print(game_controller.snakes[0])

    def select_action(self, snake_object):
        """
        TODO: select an action with a probability of EPSILON of choosing max Q-value
        :return: an action index
        """
        snake_dir = snake_object.direction
        snakex = snake_object.head[0]
        snakey = snake_object.head[1]
        # Begin with totaly random movement
        while True:
            action = self.env.action_space.sample() # take a random action
            if action == snake_object.RIGHT and snake_dir != snake_object.LEFT:
                return snake_object.RIGHT
            elif action == snake_object.LEFT and snake_dir != snake_object.RIGHT:
                return snake_object.LEFT
            elif action == snake_object.UP and snake_dir != snake_object.DOWN:
                return snake_object.UP
            elif action == snake_object.DOWN and snake_dir != snake_object.UP:
                return snake_object.DOWN
            else:
                # Loop to try another action
                pass



    def play(self):
        for i_episode in range(self.EPISODES):
            observation = self.env.reset()
            
            game_controller = self.env.controller
            snakes_array = game_controller.snakes
            snake_object1 = snakes_array[0]
            grid_object = game_controller.grid
            #coord_food = get_food_location(grid_object, game_controller)
            #print("food located at {}".format(coord_food))

            prev_action = 0
            for t in range(100):
                self.env.render()
                print("interval t={}".format(t))
                #print(observation)
                action = self.select_action(snake_object1)
                #prev_action = action
                if (action == None):
                    break
                observation, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

        self.env.close()
                



if __name__ == "__main__":
    app = SnakeQlearning()
    app.play()

