# Uses the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a shortest path version
# Contains a method that find the food
# Note that this only works for one food goal

# TODO: look into GoalEnv

import gym
import gym_snake


EPISODES = 20  # number of episodes


def select_action(snake_object, coord_food):
    """
    This selects a move that can be likened to shortest path (almost)
    :return: an action index
    """
    snake_dir = snake_object.direction
    snakex = snake_object.head[0]
    snakey = snake_object.head[1]
    if snakex < coord_food[0] and snake_dir != snake_object.LEFT:
        return snake_object.RIGHT
    elif snakex > coord_food[0] and snake_dir != snake_object.RIGHT:
        return snake_object.LEFT
    elif snakey > coord_food[1] and snake_dir != snake_object.DOWN:
        return snake_object.UP
    elif snakey < coord_food[1] and snake_dir != snake_object.UP:
        return snake_object.DOWN
    else:
        # todo: handle second best movement in case snake is pointing in wrong direction
        return None


def get_food_location(grid_object, controller):
    """
    Finds the location of a single food by looping
    over the grid and checking if pixel color is FOOD_COLOR
    :return: coord
    """
    #grid_pixels = grid_object.grid
    for x in range(grid_object.grid_size[1]):
        for y in range(grid_object.grid_size[0]):
            coord = [x,y]
            if controller.grid.food_space(coord):
                return coord
    return None


# Begin
env = gym.make('snake-v0')
env.n_foods = 1
env.random_init = False

##env.reset()
##game_controller = env.controller
##print(game_controller.snakes[0])


for i_episode in range(EPISODES):
    observation = env.reset()
    
    game_controller = env.controller
    snakes_array = game_controller.snakes
    snake_object1 = snakes_array[0]
    grid_object = game_controller.grid
    coord_food = get_food_location(grid_object, game_controller)
    print("food located at {}".format(coord_food))

    prev_action = 0
    for t in range(100):
        env.render()
        print("interval t={}".format(t))
        #print(observation)
        action = select_action(snake_object1, coord_food)
        #prev_action = action
        if (action == None):
            break
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
        
