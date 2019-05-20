# Uses a modified verson of the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a blind version Q-learning version
# Note that this only works for one food goal
# In this version the snake starting position is random
# In this version the trophy position is controlled

import gym
import gym_snake
import numpy as np
from pickle import loads, dump
from random import randint, random
import time

class SnakeQlearning:
    """
    Snake game that uses Q-learning.
    All coordinates are in x,y order (col,row/width,height)
    Author: Johan Alfredeen
    Date: 2019-05-19
    """

    ETA = 0.1  # learning rate
    GAMMA = 0.95  # discount
    STARTING_EPSILON = 0.8  # exploration: greedy - random, epsilon value at beginning
    MIN_EPSILON = 0.2  # minimum exploration (toward end of training)
    EPISODES = 10000  # number of episodes
    MAX_EPISODE_TIMESTEPS = 500  # max number of timesteps (action moves) per episode
    REWARD = 0   # default reward (if negative, small penalty for every step)
    FRAME_SPEED = 0.0001  # the frame speed for rendering (lower is faster)
    TROPHY_POS = [2, 3]  # the trophy pos at initialization
    DEBUG = False  # The verbosity level for logging

    def __init__(self, fixed_snake=False, display=True):
        """
        Class Initializer
        :param fixed_snake: controlls whether the snake startin position is fixed or random.
        """
        self.env = gym.make('snake-mod19-v0')
        self.env.n_foods = 1
        self.env.grid_size = [10,10]
        self.env.snake_size = 2
        self.env.random_init = False # If set to false, the food units initialize to the same location at each reset
        self.fixed_snake = fixed_snake
        self.display = display
        self.frame_speed = self.FRAME_SPEED
        self.epsilon = self.STARTING_EPSILON

        self.snake = None
        self.trophy_pos = self.TROPHY_POS
        self.width = self.env.grid_size[0]
        self.height = self.env.grid_size[1]
        self.log = []
        self.q = Qlearn(self.width, self.height, self.ETA, self.GAMMA, self.epsilon, 0)
        self.gl_metrics = dict(snakestart=[-1,-1], trophy=[-1,-1],
            disttotrophy=-1, beststepstrophy=-1, avgrewardsperstep=0, num_seq_episodes_success=0)

    def select_action(self):
        """
        Select an action with a probability of epsilon of choosing max Q-value
        :return: an action index
        """
        if randint(0,100) < self.epsilon * 100:
            # Explore
            # select a random direction including the max
            idx = randint(0,3) 
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

    def train(self):
        """
        Trains a single game level of snake.
        :return: the number of episodes run
        """
        msg = str.format("Start run of snake_qlearn_mod19 at {0}, using eta:{1}, gamma:{2}, epsilons:{3} {4}", \
            time.strftime('%m-%d-%Y %H:%M:%S'), self.ETA, self.GAMMA, self.STARTING_EPSILON, self.MIN_EPSILON)
        self._log(msg, screen=True)

        # TODO: to support multiple game levels: implement new food positions, save to file
        self.gl_metrics['num_seq_episodes_success'] = 0 # the number of sequential episodes with success (find the trophy)
        for epi in range(1, self.EPISODES+1):
            if self.display and epi > self.EPISODES*0.9:
                display = True
            else:
                display = False
            success = self.train_episode(display)
            if success == 0:
                self.gl_metrics['num_seq_episodes_success'] = 0
            else:
                self.gl_metrics['num_seq_episodes_success'] += 1

            if self.gl_metrics['num_seq_episodes_success'] > 50 and self.q.check_all_states_nonzero():
                # Consider training done, converged
                self._log("Completed training for this game level. Reached nr seq episodes.", True)
                break

            if self.epsilon>self.MIN_EPSILON and epi>=(self.EPISODES*0.2) and epi%10 == 0:
                # epsilon decay after the first 20pct episodes, reach min after about 75%
                self.epsilon -= 10*(self.STARTING_EPSILON-self.MIN_EPSILON)/(self.EPISODES*0.55)
                if self.epsilon < self.MIN_EPSILON:
                    self.epsilon = self.MIN_EPSILON

            self._log("Episode:{0}, success:{1}, numseqsuccesses:{2}, epsilon:{3}"
                .format(epi, success, self.gl_metrics['num_seq_episodes_success'], round(self.epsilon,4)), True)
        self.env.close()
        self._log(str.format("Finished RL Q-learning with metrics:{0}",self.gl_metrics), screen=True)
        self._log(str.format("num_non_zero_states:{0}", self.q.get_num_nonzero_states()))
        return(epi)

    def get_new_snake_start_coord(self):
        """
        Gets a new valid starting coordinate for the snake
        :return: a new coordinate
        """
        # TODO: unit test
        snake_coord = [randint(0, self.width-1), randint(0, self.height-1)]
        while snake_coord == self.trophy_pos:
            snake_coord = [randint(0, self.width-1), randint(0, self.height-1)]
        return snake_coord

    def train_episode(self, display):
        """
        Trains a single episodes of game snake.
        An episode ends in one of following: find trophy, make invalid move/die, reach max timesteps
        :return: boolean success found trophy or failure
        """
        self.env.food_pos = self.trophy_pos

        if not self.fixed_snake:
            self.env.start_coord = self.get_new_snake_start_coord()

        observation = self.env.reset()  # observation contains color
        game_controller = self.env.controller
        self.snake = game_controller.snakes[0]

        self._log("Reset. Food position:{0}, snake head position {1}, previous start coord:{2}"
            .format(game_controller.grid.food_pos, self.snake.head, self.gl_metrics['snakestart']), True)

        if not np.array_equal(game_controller.grid.food_pos, self.trophy_pos):
            self._log("WARNING !!!  env food position is not equal to trophy position. Ending this episode.", True)
            return 0

        if not self.fixed_snake:
            assert np.array_equal(self.snake.head, self.env.start_coord)

        self.gl_metrics['trophy'] = self.trophy_pos
        self.gl_metrics['snakestart'] = self.snake.head

        state = self.snake.head # state is coord location of snake
        totalreward = 0

        for t in range(1, self.MAX_EPISODE_TIMESTEPS+1):
            
            if display:
                self.env.render(frame_speed=self.frame_speed)
            
            action = self.select_action() # select action from Q matrix
            if (action is None):
                #break
                raise Exception("Do we ever enter here? action is None")
            observation, reward, done, _ = self.env.step(action)
            if reward == 0:
                reward = self.REWARD
            self._log_debug("reward:{0}, done:{1}".format(reward, done))
            nxt_state = self.snake.head
            #self._log_debug("next state:{0}".format(nxt_state))
            totalreward += reward
            action_idx = self.convert_action_to_index(action)
            if not self.q.is_valid_coord(state):
                self._log_debug(done)
                self._log_debug(observation)
                self._log_debug(state)
                raise Exception("state is an invalid coord, why here?")
            if not self.q.is_valid_coord(nxt_state):
                # The action moved the snake out of bounds
                # Because the env sets done to true, set the q-value to -1 and run new episode
                #self.q.set_qvalue(-1, state, action_idx)
                self.q.update_state(reward, state, action_idx, nxt_state) # update the Q matrix
                return 0
            self.q.update_state(reward, state, action_idx, nxt_state) # update the Q matrix
            state = nxt_state
            if done:
                self._log_debug("Episode finished after {} timesteps. done=True".format(t+1))
                if reward == 1:
                    print("FOUND THE FRUIT, done")
                    return 1
                return 0

            # We finish the episode if we found the fruit
            if reward == 1:
                # Note that the env does not consider finding fruit to be done
                # TODO: must solve the moving fruit!
                # Every time we eat the fruit the fruit moves
                return 1

        self._log_debug("Finished episode with total accumulated reward = {0}".format(totalreward))
        return 0

    def play_minigame(self):
        # Simple game to test various game parameters
        observation = self.env.reset()  # observation contains color
        game_controller = self.env.controller
        self.snake = game_controller.snakes[0]
        for t in range(2):
            print("timestep t={}".format(t))
            if self.display:
                self.env.render(frame_speed=self.frame_speed)
        self.env.close()

    def play_initgame(self):
        # Used by unit testing
        observation = self.env.reset()  # observation contains color
        game_controller = self.env.controller
        self.snake = game_controller.snakes[0]
        self.env.close()

    def replay(self):
        """
        Replays previously trained snake from file
        """
        pass
        #with open(self.file, 'rb') as handle:
        #   self.q.qmap = loads(handle.read())

    def save_qdata(self, path):
        with open(path, 'wb') as file:
            dump(self.q.qmap, file)

    def save_log(self, path):
        with open(path, 'w') as file:
            for msg in self.log:
                file.write(msg + '\n')

    def _log(self, msg, screen=False, is_debug=False):
        if self.DEBUG or not is_debug:
            self.log.append(msg)
            if screen:
                print(msg)

    def _log_debug(self, msg):
        self._log(msg, screen=False, is_debug=True)



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
                if not np.array_equal(self.qmap[i][j], [0,0,0,0]):
                    num += 1
        return num

    def check_all_states_nonzero(self):
        return app.q.get_num_nonzero_states() == app.width*app.height - 1

    def get_optimal_path(self, startcoord, trophycoord):
        """
        Determine the optimal path per game episode given a starting state
        :return: an ordered list of states (coord)
        """
        assert startcoord != trophycoord
        p=[]
        p.append(startcoord)
        coord=startcoord
        i=0
        while True:
            coord=self.get_coord_next_max(coord)
            if not self.is_valid_coord(coord):
                raise Exception(str.format("No optimal path exists for coord {0}. \
                    Invalid next coordinate in get_optimal_path for coord:{1}, vals:{2}", \
                    startcoord, p[i], self.get_statevalues(p[i][0], p[i][1])))
            if i>1 and coord==p[i-1]:
                raise Exception(str.format("No optimal path exists for coord {0}. Invalid path (circular). Incomplete path:{1}",startcoord,p))
            p.append(coord)
            if np.array_equal(coord, trophycoord):
                return p
            i+=1
            if i>self.w*self.h:
                raise Exception("No optimal path exists in this q-table for coord {0}".format(startcoord))
        return p

    def verify_all_states(self, trophy_pos):
        """
        Verifies that there exists a valid path from every state to the trophy
        We skip the wall states
        :return: an array containing any states with invalid paths
        """
        states = []
        coords = self.get_play_coords(trophy_pos)
        for i in range(len(states)):
            coord = coords[i]
            try:
                path = self.get_optimal_path(coord, trophy_pos)
                print("Optimal path for {0} is {2}".format(coord, path))
            except Exception as e:
                states.append(coord)
                print(e)
        return states

    def get_play_coords(self, trophy_pos):
        """
        Gets all possible play coordinates for the snake.
        Excludes the trophy state and all wall coordinates
        :return: an array of coordinates
        """
        coords = []
        for i in range(1,self.h-1):
            for j in range(1,self.w-1):
                if not np.array_equal([j,i], trophy_pos):
                    coords.append([j,i])
        return coords

            


if __name__ == "__main__":
    start = time.time()
    app = SnakeQlearning(fixed_snake=False, display=False)
    episodes = app.train()
    end = time.time()
    file_name = str.format("rl_gymsnake_mod19_{0}.log", time.strftime('%Y%m%d_%H%M%S'))
    app.save_log(str.format("C:/Dev/logs/{0}", file_name))

    #print("Q-values")
    app.display_qvalues()
    qpath = 'C:/Dev/logs/rl_snake_qdata.txt'
    #app.save_qdata(qpath)
    #app.display_gl_metrics()
    #print("num_non_zero_states:{0}".format(app.q.get_num_nonzero_states()))

    if not app.q.check_all_states_nonzero():
        print("Training did not succeed. All states are not non zero.")

    if app.q.check_all_states_nonzero():
        print("Now verifying the training using all states")    
        try:
            invalid_states = app.q.verify_all_states(app.gl_metrics['trophy'])
            print("The number of states with invalid paths:{0}. {1}".format(len(invalid_states), invalid_states))
        except Exception as e:
            print("Unable to determine best path from snake to trophy.")
            print(e)

        print("Finished game level after {0} training episodes".format(episodes))
        print("Total train duration:{0} seconds" .format(round(end - start, 3)))

