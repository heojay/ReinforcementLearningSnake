# Uses the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a block distance version / shortest path
# Contains a method that find the food

# TODO: look into GoalEnv

import gym
import gym_snake


def select_action(prev_action):
    """
    This simply moves in a circle
    :return: an action index
    """
    if prev_action == 0:
        return 1
    elif prev_action == 1:
        return 2
    elif prev_action == 2:
        return 3
    elif prev_action == 3:
        return 0

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


for i_episode in range(20):
    observation = env.reset()
    
    game_controller = env.controller
    grid_object = game_controller.grid
    coord = get_food_location(grid_object, game_controller)
    print("food located at {}".format(coord))

    prev_action = 0
    for t in range(10):
        env.render()
        print("interval t={}".format(t))
        #print(observation)
        action = select_action(prev_action)
        prev_action = action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
        
