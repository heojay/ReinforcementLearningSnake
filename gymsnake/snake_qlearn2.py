# Uses the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a blind version Q-learning version
# Note that this only works for one food goal
# In this version the snake starting location is fixed

import gym
import gym_snake
import numpy as np
import random
import time

class SnakeQlearning:
    """
    Snake game that uses Q-learning.
    All coordinates are in x,y order (col,row/width,height)
    Author: Johan Alfredeen
    Date: 2019-05-06
    """

    ETA = 0.2  # learning rate
    GAMMA = 0.95  # discount
    STARTING_EPSILON = 0.9  # exploration: greedy - random, epsilon value at beginning
    MIN_EPSILON = 0.2       # minimum exploration (toward end of training)
    REWARD = 0  # default reward # TODO: not used
    EPISODES = 100  # number of episodes
    MAX_EPISODE_TIMESTEPS = 500 # max number of timesteps (action moves) per episode
    FRAME_SPEED = 0.0001 # the frame speed for rendering (lower is faster)

    def __init__(self):
        """
        Class Initializer
        """
        self.env = gym.make('snake-v0')
        self.env.n_foods = 1
        self.env.grid_size = [10,10]
        self.env.snake_size = 2
        #self.env.unit_size = 4
        #self.env.unit_gap = 1
        self.env.random_init = False # If set to false, the food units initialize to the same location at each reset
        self.frame_speed = self.FRAME_SPEED
        self.epsilon = self.STARTING_EPSILON

        self.snake = None
        self.width = self.env.grid_size[0]
        self.height = self.env.grid_size[1]
        self.q = Qlearn(self.width, self.height, self.ETA, self.GAMMA, self.epsilon, self.REWARD)
        self.gl_metrics = dict(snakestart=[-1,-1], trophy=[-1,-1],
            disttotrophy=-1, beststepstrophy=-1, avgrewardsperstep=0, num_seq_episodes_success=0)

    def select_action(self):
        """
        Select an action with a probability of epsilon of choosing max Q-value
        :return: an action index
        """
        if random.randint(0,100) < self.epsilon * 100:
            # Explore
            # select a random direction including the max
            idx = random.randint(0,3) 
        else:
            # use the direction of the max-value
            idx = self.q.get_index_max_value(self.snake.head[0], self.snake.head[1])
        return self.convert_idx_to_action(idx)

    def convert_idx_to_action(self, idx):
        if idx == 0:
            return self.snake.UP
        elif idx == 1:
            return self.snake.RIGHT
        elif idx == 2:
            return self.snake.DOWN
        elif idx == 3:
            return self.snake.LEFT

    def convert_action_to_index(self, action):
        if action == self.snake.UP:
            return 0
        elif action == self.snake.RIGHT:
            return 1
        elif action == self.snake.DOWN:
            return 2
        elif action == self.snake.LEFT:
            return 3

    def is_legal_action(self, action):
        """
        Checks if a given action is legal based on direction of snake
        :return: True or False
        """
        dir = self.snake.direction
        if action == self.snake.RIGHT and dir == self.snake.LEFT:
            return False
        elif action == self.snake.LEFT and dir == self.snake.RIGHT:
            return False
        elif action == self.snake.UP and dir == self.snake.DOWN:
            return False
        elif action == self.snake.DOWN and dir == self.snake.UP:
            return False
        else:
            return True

    def display_qvalues(self):
        print(self.q.qmap)

    def display_gl_metrics(self):
        print(self.gl_metrics)

    def play_initgame(self):
        # Used by unit testing
        observation = self.env.reset()  # observation contains color
        game_controller = self.env.controller
        self.snake = game_controller.snakes[0]
        self.env.close()

    def play(self):
        """
        Plays a single game level of snake.
        :return: the number of episodes run
        """
        self.gl_metrics['num_seq_episodes_success'] = 0 # the number of sequential episodes with success (find the trophy)
        for epi in range(1, self.EPISODES+1):
            # todo: could put a try-catch here
            success = self.play_episode()
            if success == 0:
                self.gl_metrics['num_seq_episodes_success'] = 0
            else:
                self.gl_metrics['num_seq_episodes_success'] += 1

            if self.gl_metrics['num_seq_episodes_success'] > 20:
                # Consider training done, converged
                print("Completed training for this game level. Reached nr seq episodes.")
                break

            if self.epsilon>self.MIN_EPSILON and epi>(self.EPISODES*0.1) and epi%10 == 0:
                # epsilon decay after the first 10pct episodes
                self.epsilon -= 10*(self.STARTING_EPSILON-self.MIN_EPSILON)/(self.EPISODES*0.9)

            print("Episode:{0}, success:{1}, numseqsuccesses:{2}, epsilon:{3}"
                .format(epi, success, self.gl_metrics['num_seq_episodes_success'], self.epsilon))
        self.env.close()
        return(epi)

    def play_episode(self):        
        """
        Plays a single episodes of game snake.
        An episode ends in one of following: find trophy, make invalid move/die, reach max timesteps
        :return: boolean success found trophy or failure
        """
        observation = self.env.reset()  # observation contains color
        game_controller = self.env.controller
        self.snake = game_controller.snakes[0]

        # Here we would like to position the snake in a new random location,
        # but this does not seem to be possible with this snake gym version.

        if np.array_equal(self.gl_metrics['snakestart'], [-1,-1]):
            self.gl_metrics['snakestart'] = self.snake.head
        elif np.array_equal(self.gl_metrics['snakestart'], self.snake.head) == False:
            raise Exception("Snake starting location is not same as before!")

        state = self.snake.head # state is coord location of snake
        #print("state:{0}".format(state))

        metrics = self.gl_metrics
        totalreward = 0

        for t in range(1, self.MAX_EPISODE_TIMESTEPS+1):
            self.env.render(frame_speed=self.frame_speed)
            action = self.select_action() # select action from Q matrix
            if (action == None):
                #break
                raise Exception("Do we ever enter here? action == None")
            observation, reward, done, _ = self.env.step(action)
            #print("reward:{0}, done:{1}".format(reward, done))
            nxt_state = self.snake.head
            #print("next state:{0}".format(nxt_state))
            totalreward += reward
            action_idx = self.convert_action_to_index(action)
            if self.q.is_valid_coord(state) == False:
                print(done)
                print(observation)
                print(state)
                raise Exception("state is an invalid coord, why here?")
            if self.q.is_valid_coord(nxt_state) == False:
                # The action moved the snake out of bounds
                # Because the env sets done to true, set the q-value to -1 and run new episode
                self.q.set_qvalue(-1, state, action_idx)
                return 0
            self.q.update_state(reward, state, action_idx, nxt_state) # update the Q matrix
            state = nxt_state
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                if reward == 1:
                    print("FOUND THE FRUIT, done")
                    self.gl_metrics['trophy']=self.snake.head
                    return 1
                    #self.env.reset()
                return 0

            # We finish the episode if we found the fruit
            if reward == 1:
                # Note that the env does not consider finding fruit to be done
                # TODO: must solve the moving fruit!
                # Every time we eat the fruit the fruit moves
                self.gl_metrics['trophy']=self.snake.head
                return 1

        print("Finished episode with total accumulated reward = {0}".format(totalreward))
        return 0

    def play_minigame(self):
        # Simple game to test various game parameters
        observation = self.env.reset()  # observation contains color
        game_controller = self.env.controller
        self.snake = game_controller.snakes[0]
        for t in range(2):
                print("timestep t={}".format(t))
                self.env.render(frame_speed=self.frame_speed)
        self.env.close()


class Qlearn:
    """
    Implementation of Q-learning
    Holds Q-values in a matrix with tuples of 4 values
    Date: 2019-05-06
    """

    lr = 0.1 # alpha learn_rate
    disc = 0.95  # gamma discount
    expl = 1.0 # exploration
    reward = 0 # default_reward

    def __init__(self, width, height, lr, disc, expl, reward):
        self.w = width
        self.h = height
        self.lr = lr
        self.disc = disc
        self.expl = expl
        self.reward = reward
        self.qmap = [[[0] * 4 for _ in range(height)]
                       for _ in range(width)]  # matrix of states: N E S W, init to 0 values

    def get_statevalues(self, col, row):
        return self.qmap[col][row]

    def get_index_max_value(self, col, row):
        """
        :return: the index (N E S W) of the max value of the state cell
        """
        values=self.get_statevalues(col, row)
        return values.index(max(values))

    def update_state(self, reward, state, action_idx, nxt_state):
        """
        Updates the Q-value of a state
        """
        qv = self.get_update_qvalue(reward, state, action_idx, nxt_state)
        self.set_qvalue(qv, state, action_idx)

    def set_qvalue(self, qvalue, state, action_idx):
        self.qmap[state[0]][state[1]][action_idx] = qvalue

    def get_update_qvalue(self, reward, state, action_idx, nxt_state):
        """
        Calculates a new Q-value for a state
        :return: the new Q-value
        """
        #print("state:{0}".format(state))
        #print("nxt state:{0}".format(nxt_state))
        oldval = self.qmap[state[0]][state[1]][action_idx]
        #print("oldval={0}".format(oldval))
        if self.is_valid_coord(nxt_state):
            nxt_vals = self.qmap[nxt_state[0]][nxt_state[1]]
            nxt_max = max(nxt_vals)
            #print("max nxt={0}".format(nxt_max))
        else:
            nxt_max = -1

        learnedval = (reward + self.disc * nxt_max ) # nxt_state should be coord
        #print("learnedval={0}".format(learnedval))
        qv = (1-self.lr) * oldval + self.lr * learnedval
        #print("Q-value={0}".format(qv))
        return qv

    def get_coord_next_max(self, coord):
        """
        Gets the coord in the direction of the next max q-value
        :return: a coord of x,y
        """
        col=coord[0]
        row=coord[1]
        maxidx=self.get_index_max_value(col,row)
        if maxidx==0:
            return [col,row-1] # north
        elif maxidx==1:
            return [col+1,row] # east
        elif maxidx==2:
            return [col,row+1] # south
        elif maxidx==3:
            return [col-1,row] # west
        else:
            raise Exception("Invalid q-value index in get_coord_next_max (not in 0:3)")

    def is_valid_coord(self, coord):
        if coord[0] < 0 or coord[0] > self.w-1:
            return False
        if coord[1] < 0 or coord[1] > self.h-1:
            return False
        return True

    def get_num_nonzero_states(self):
        """
        Counts the number of states that are not entirely 0
        :return: the number of non-zero states
        """
        num=0
        for i in range(self.h):
            for j in range(self.w):
                if np.array_equal(self.qmap[i][j], [0,0,0,0]) == False:
                    num += 1
        return num

    def get_optimal_path(self, startcoord, trophycoord):
        """
        Determine the optimal path per game episode given a starting state
        :return: an ordered list of states (coord)
        """
        p=[]
        p.append(startcoord)
        coord=startcoord
        i=0
        while True:
            coord=self.get_coord_next_max(coord)
            if self.is_valid_coord(coord) == False:
                raise Exception("No optimal path exists. Invalid next coordinate in get_optimal_path")
            if i>1 and coord==p[i-1]:
                raise Exception("No optimal path exists. Invalid path (circular)")
            p.append(coord)
            if np.array_equal(coord, trophycoord):
                return p
            i+=1
            if i>self.w*self.h:
                raise Exception("No optimal path exists in this q-table for starting coord {0}".format(startcoord))
        return p




if __name__ == "__main__":
    start = time.time()
    app = SnakeQlearning()
    episodes = app.play()
    end = time.time()
    #print("Q-values")
    #app.display_qvalues()
    app.display_gl_metrics()
    print("num_non_zero_states:{0}".format(app.q.get_num_nonzero_states()))
    try:
        path = app.q.get_optimal_path(app.gl_metrics['snakestart'], app.gl_metrics['trophy'])
        print("Optimal path 1:{0}".format(path))
        path = app.q.get_optimal_path([0,0], app.gl_metrics['trophy'])
        print("Optimal path 2:{0}".format(path))
        path = app.q.get_optimal_path([10,10], app.gl_metrics['trophy'])
        print("Optimal path 3:{0}".format(path))
    except Exception as e:
        print("Unable to determine best path from snake to trophy.")
        print(e)

    print("Finished game level after {0} training episodes".format(episodes))
    print("Total play duration:{0} seconds" .format(end - start))    

