#!/usr/bin/env python3

# Uses a modified verson of the OpenAI gym from
#   https://github.com/grantsrb/Gym-Snake
# This is a blind version Q-learning version
# This version workds for mutiple game levels
# In this version the snake starting position is random
# In this version the trophy positions are pre-set

import gym
import gym_snake
import numpy as np
import time
import matplotlib.pyplot as plt
from pickle import loads, dump
from random import randint, random


class SnakeQlearning:
    """
    Snake game that uses Q-learning.
    All coordinates are in x,y order (col,row/width,height)
    Author: Johan Alfredeen
    Date: 2019-05-19
    """

    ETA = 0.1  # learning rate
    GAMMA = 0.99  # discount
    STARTING_EPSILON = 0.9  # exploration: greedy - random, epsilon value at beginning
    MIN_EPSILON = 0.6  # minimum exploration (toward end of training)
    EPISODES = 10000  # initial maximum number of episodes to train per game level
    MAX_EPISODE_TIMESTEPS = 500  # max number of timesteps (action moves) per episode
    REWARD = 0   # default reward (if negative, small penalty for every step)
    FRAME_SPEED = 0.0001  # the frame speed for rendering (lower is faster)
    TROPHY_POSITIONS = [[7, 7], [2, 3], [4,8], [3,12]] # trophy positions at each game level
    SNAKE_SIZE = 2  # the initial length of the snake at level 1
    DEBUG = False  # The verbosity level for logging

    def __init__(self, fixed_snake=False, display=True, n_levels=1):
        """
        Class Initializer
        :param fixed_snake: controls whether the snake starting position is fixed or random.
        :param display: when true, then displays the snake's movements at the end of training.
        :param n_levels: the number game levels for training and replay
        """
        assert n_levels == 1 or len(self.TROPHY_POSITIONS) > 1

        self.env = gym.make('snake-mod19-v0')
        self.env.n_foods = 1
        self.env.grid_size = [15,15]
        self.env.random_init = False # If set to false, the food units initialize to the same location at each reset
        self.env.snake_size = self.SNAKE_SIZE
        self.fixed_snake = fixed_snake
        self.display = display # If display snake game, then training score plot is not displayed
        self.score_freq = self.EPISODES / 10

        self.level = 1
        self.trophy_pos = self.TROPHY_POSITIONS[0]
        self.n_game_levels = n_levels
        self.width = self.env.grid_size[0]
        self.height = self.env.grid_size[1]
        self.log = []
        self.q = Qlearn(self.width, self.height, self.ETA, self.GAMMA)
        self._reinitialize()

    def _reinitialize(self):
        self.EPISODES += (2000*self.level)
        self.snake = None
        self.epsilon = self.STARTING_EPSILON
        self.score_per_episode = []
        self.score_per_episode_x = []
        self.n_invalid_paths_per_episode = []
        self.gl_metrics = dict(snakestart=[-1,-1], trophy=[-1,-1], disttotrophy=-1, 
            num_seq_episodes_success=0, last_score=0, last_n_invalid_paths=-1)

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

    def train(self, qfile=None, from_level=1):
        """
        Trains the snake game using Q-learning
        :param qfile: path and filename for saving Q-data or None if not to save
        :param from_level: the beginning level to train from
        :return: the number of game levels successfully trained
        """
        assert from_level > 0 and from_level <= self.n_game_levels

        for level in range(from_level, self.n_game_levels+1):
            self.level = level
            self._log("Begin training game level {0}".format(level), True)
            if level == 1:
                req_score = 90
            if level > 1:
                self._reinitialize()
                self.trophy_pos = self.get_level_trophy(self.level)
                self._log(str.format("Set trophy position to {0}", self.trophy_pos))
                self.env.snake_size = self.SNAKE_SIZE + level - 1
                req_score = 75

            if qfile is None:
                qfilelev = None
            else:
                qfilelev = str.format("{0}_{1}.txt", qfile, level)

            self.env.food_pos = self.trophy_pos
            self.train_game_level(qfilelev, req_score)
        return level

    def train_game_level(self, qfile=None, req_score=90):
        """
        Trains a single game level of snake.
        :param qfile: path and filename for saving Q-data or None if not to save
        :param req_score: the minimum score accepted for early stopping
        :return: the number of episodes run
        """
        msg = str.format("{0}: Start run of snake_qlearn, using eta:{1}, gamma:{2}, epsilons:{3} {4}", \
            time.strftime('%Y-%m-%d %H:%M:%S'), self.ETA, self.GAMMA, self.STARTING_EPSILON, self.MIN_EPSILON)
        self._log(msg, screen=True)

        self.gl_metrics['num_seq_episodes_success'] = 0 # the number of sequential episodes with success (find the trophy)
        for epi in range(1, self.EPISODES+1):
            if self.display and epi>self.EPISODES*0.9:
                display = True
            else:
                display = False
            success = self.train_episode(display)
            if success == 0:
                self.gl_metrics['num_seq_episodes_success'] = 0
            else:
                self.gl_metrics['num_seq_episodes_success'] += 1

            if epi%self.score_freq == 0:
                try:
                    score = round(self.q.calc_current_score(self.trophy_pos), 3)
                except:
                    score = -1
                self.gl_metrics['last_score'] = score
                self.score_per_episode.append(score)
                self.score_per_episode_x.append(epi)
                n_invalid_paths = len(self.q.verify_all_states(self.trophy_pos))
                self.gl_metrics['last_n_invalid_paths'] = n_invalid_paths
                self.n_invalid_paths_per_episode.append(n_invalid_paths)

                if self.level == 1:
                    if score>req_score and n_invalid_paths==0 and epi<self.EPISODES:
                        # Early stopping
                        self._log("Completed training for this game level. All states are trained.", True)
                        break
                else:
                    # Can we stop early, given the growing tail?
                    try:
                        if len(self.q.get_optimal_path(self.env.start_coord,self.trophy_pos)) == self.q._calc_shortest_dist(self.env.start_coord,self.trophy_pos):
                            # Early stopping
                            self._log("Completed training for this game level. The snake start is trained optimally.", True)
                            break
                    except:
                        pass

            if self.epsilon>self.MIN_EPSILON and epi>=(self.EPISODES*0.2) and epi%10 == 0:
                # epsilon decay after the first 20pct episodes, reach min after about 75%
                self.epsilon -= 10*(self.STARTING_EPSILON-self.MIN_EPSILON)/(self.EPISODES*0.55)
                if self.epsilon < self.MIN_EPSILON:
                    self.epsilon = self.MIN_EPSILON

            self._log("Level:{0}, Episode:{1}, success:{2}, numseqsuccesses:{3}, last score:{4}, last nrinvalidpaths:{5} epsilon:{6}"
                .format(self.level, epi, success, self.gl_metrics['num_seq_episodes_success'],
                    self.gl_metrics['last_score'], self.gl_metrics['last_n_invalid_paths'], round(self.epsilon,4)), True)

        self.env.close()
        self._log(str.format("Finished RL Q-learning with metrics:{0}",self.gl_metrics), screen=True)
        self._log(str.format("num_non_zero_states:{0}", self.q.get_num_nonzero_states()))

        if self.level == 1:
            n_invalid_paths = len(self.q.verify_all_states(self.trophy_pos))
            if n_invalid_paths > 0:
                self._log(str.format("ERROR: Unable to continue training. Level {0} failed to train all states:{1}", self.level, n_invalid_paths))
                #raise Exception(str.format("Unable to continue training. Level {0} failed to train all states:{1}", self.level, n_invalid_paths))
        else:
            try:
                self.q.get_optimal_path(self.env.start_coord, self.trophy_pos)
            except:
                self._log(str.format("ERROR: Unable to continue training. Level {0} failed to train the snake start state:{1}", self.level, self.env.start_coord))
                #raise Exception(str.format("Unable to continue training. Level {0} failed to train the snake start state:{1}", self.level, self.env.start_coord))

        if qfile is not None:
            self.save_qdata(qfile)

        return(epi)

    def train_episode(self, display):
        """
        Trains a single episode of game snake.
        An episode ends in one of following: find trophy, make invalid move/die, reach max timesteps
        :param display: whether to render the snake in the game map during training
        :return: boolean success found trophy or failure
        """
        if not self.fixed_snake:
            if self.level == 1:
                self.env.start_coord = self.get_new_snake_start_coord()
            else:
                # get the position of trophy from previous level
                self.env.start_coord = self.get_level_trophy(self.level-1)  #self.TROPHY_POSITIONS[self.level-2]

        observation = self.env.reset()
        game_controller = self.env.controller
        game_controller.grid.food_pos = self.env.food_pos
        self.snake = game_controller.snakes[0]

        self.gl_metrics['trophy'] = self.trophy_pos
        self.gl_metrics['snakestart'] = self.snake.head

        self._log("Reset. Food:{0}, snake head:{1}, previous start:{2}"
            .format(game_controller.grid.food_pos, self.snake.head, self.gl_metrics['snakestart']), screen=False)

        if not np.array_equal(game_controller.grid.food_pos, self.trophy_pos):
            self._log("WARNING !!!  env food position is not equal to trophy position. Ending this episode.", True)
            return 0

        if not self.fixed_snake:
            assert np.array_equal(self.snake.head, self.env.start_coord)
        state = self.snake.head # state is coord location of snake
        totalreward = 0

        for t in range(1, self.MAX_EPISODE_TIMESTEPS+1):
            
            if display:
                self.env.render(frame_speed=self.FRAME_SPEED)
            
            action = self.select_action() # select action from Q matrix
            if (action is None):
                raise Exception("Unable to perform the selected action. Action is None.")
            observation, reward, done, _ = self.env.step(action)
            if reward == 0:
                reward = self.REWARD
            self._log_debug("reward:{0}, done:{1}".format(reward, done))
            nxt_state = self.snake.head
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
                # Every time we eat the fruit the fruit moves
                return 1

        self._log_debug("Finished episode with total accumulated reward = {0}".format(totalreward))
        return 0

    def get_level_trophy(self, level):
        """
        Gets the trophy for a given level
        """
        trophy_idx = (level-1)%len(self.TROPHY_POSITIONS)
        return self.TROPHY_POSITIONS[trophy_idx]

    def get_new_snake_start_coord(self, exclude_wall_states=False):
        """
        Gets a new valid starting coordinate for the snake
        :param exclude_wall_states: boolean whether to exclude the wall states
        :return: a new coordinate
        """
        if exclude_wall_states:
            w1, w2, h1, h2 = 1, self.width-2, 1, self.height-2
        else:
            w1, w2, h1, h2 = 0, self.width-1, 0, self.height-1
        snake_coord = [randint(w1, w2), randint(h1, h2)]
        while snake_coord == self.trophy_pos:
            snake_coord = [randint(w1, w2), randint(h1, h2)]
        
        return snake_coord

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
        if abs(action - self.snake.direction) == 2:
            return False
        return True

    def display_qvalues(self):
        print(self.q.qmap)

    def display_gl_metrics(self):
        print(self.gl_metrics)

    def play_minigame(self):
        """
        Plays a simple game to test various game parameters
        """
        observation = self.env.reset()
        game_controller = self.env.controller
        self.snake = game_controller.snakes[0]
        for t in range(2):
            print("timestep t={}".format(t))
            if self.display:
                self.env.render(frame_speed=self.FRAME_SPEED)
        self.env.close()

    def play_initgame(self):
        """
        Used by unit testing
        """
        observation = self.env.reset()
        game_controller = self.env.controller
        self.snake = game_controller.snakes[0]
        self.env.close()

    def replay(self, path, frame_speed=0.2):
        """
        Replays all previously defined and trained game levels trained from file(s)
        :param path: path and filename of the file with q values
        :param frame_speed: the frame speed to render the paths
        """
        for level in range(1, self.n_game_levels+1):
            if level==1:
                dist=0
                while dist<3:
                    start=self.get_new_snake_start_coord(True)
                    dist=self.q._calc_shortest_dist(start, self.trophy_pos)
            else:
                start = self.get_level_trophy(level-1)  #self.TROPHY_POSITIONS[level-2] # get the position of trophy from previous level
            self.replay_level(path, [start], frame_speed, level)

    def replay_level(self, path, start_positions=[], frame_speed=0.3, level=1):
        """
        Replays previously trained snake from file
        :param path: path and filename of the file with q values
        :param start_positions: a list of starting positions, if empty then random positions
        :param frame_speed: the frame speed to render the paths
        :param level: the game level to replay
        """
        print("Replay mode, game level {0}".format(level))
        self.level = level
        path += str.format("_{0}.txt", level)
        with open(path, 'rb') as handle:
           self.q.qmap = loads(handle.read())

        assert self.q.check_all_states_nonzero()
        #assert self.q.verify_all_states(self.trophy_pos) == 0
        self.trophy_pos = self.get_level_trophy(self.level)
        self.env.food_pos = self.trophy_pos

        if len(start_positions) == 0:
            # create 2 random starting positions to test the snake
            for i in range(2):
                start = self.get_new_snake_start_coord(True)
                start_positions.append(start)

        for i in range(len(start_positions)):
            start_pos = start_positions[i]
            self.env.start_coord = start_pos
            print("Playing from snake start position: {0} to trophy position: {1}" .format(start_pos, self.trophy_pos))
            self._play_one_start(start_pos, frame_speed)

    def _play_one_start(self, start_pos, frame_speed=0.3):
        p = self.q.get_optimal_path(self.env.start_coord, self.trophy_pos)
        print(p)
        self.env.snake_size = self.SNAKE_SIZE + self.level - 1
        observation = self.env.reset()
        game_controller = self.env.controller
        self.snake = game_controller.snakes[0]
        self.env.render(frame_speed=frame_speed)

        pre_action=self.get_direction_to_coord(self.snake.head, p[2])
        if abs(pre_action-self.snake.direction)==2:
            #raise Exception("Snake direction is opposite of optimal path direction. Must compensate.")
            if self.snake.head[0]>1:
                self.env.step(self.snake.LEFT)
            else:
                self.env.step(self.snake.RIGHT)
            self.env.render(frame_speed=frame_speed)
            self.env.step(self.snake.UP)
            self.env.render(frame_speed=frame_speed)

        for i in range(1,len(p)):
            nxt_coord=p[i]
            action=self.get_direction_to_coord(self.snake.head, nxt_coord)
            #print("Now making move {0} to go to next coordinate {1}".format(action, nxt_coord))
            self.env.step(action)
            coord=nxt_coord
            self.env.render(frame_speed=frame_speed)
        self.env.close()

    def get_direction_to_coord(self, from_coord, to_coord):
        """
        Gets the direction from one coordinate to another
        :return: a direction
        """
        if from_coord[0] < to_coord[0]:
            return 1 # self.snake.RIGHT
        if from_coord[0] > to_coord[0]:
            return 3 # self.snake.LEFT
        if from_coord[1] < to_coord[1]:
            return 2 # self.snake.DOWN
        if from_coord[1] > to_coord[1]:
            return 0 # self.snake.UP
        else:
            raise Exception("Unable to determine correct direction to move snake.")

    def plot_training_scores(self):
        """
        Displays a graph with the resulting training score.
        """
        fig, ax1 = plt.subplots()
        fig.canvas.set_window_title('Figure RL snake training scores')
        plt.title('Training scores')
        fig.suptitle('Q-Learning')
        plt.xlabel('episode')
        
        color = 'tab:blue'
        ax1.set_ylabel('% shortest path', color=color)
        ax1.set_ylim(-10, 101)
        ax1.plot(self.score_per_episode_x, self.score_per_episode)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Nr of states with invalid path', color=color)
        ax2.set_ylim(0, self.width*self.height)
        ax2.plot(self.score_per_episode_x, self.n_invalid_paths_per_episode, color=color)
        plt.show()

    def save_qdata(self, path):
        """
        Saves the Q-values table to a file
        :param path: the path and filename to save to
        """
        with open(path, 'wb') as file:
            dump(self.q.qmap, file)

    def save_log(self, path):
        """
        Saves the log messages to a file
        :param path: the path and filename to save to
        """
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

    def __init__(self, width, height, lr, disc):
        """
        Class Initializer
        :param width: the width of the map grid
        :param height: the height of the map grid
        :param lr: the learning rate
        :param disc: the discount rate
        """
        self.w = width
        self.h = height
        self.lr = lr
        self.disc = disc
        self.n_play_coords = -1
        self.qmap = [[[0] * 4 for _ in range(height)]
                       for _ in range(width)]  # matrix of states: N E S W, init to 0 values

    def get_statevalues(self, col, row):
        """
        Gets the state values for a given state
        :return: the action values
        """
        return self.qmap[col][row]

    def get_index_max_value(self, col, row):
        """
        Gets the index of the action with the maximum Q-value
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
        """
        Sets a Q-value
        """
        self.qmap[state[0]][state[1]][action_idx] = qvalue

    def get_update_qvalue(self, reward, state, action_idx, nxt_state):
        """
        Calculates a new Q-value for a state
        :return: the new Q-value
        """
        oldval = self.qmap[state[0]][state[1]][action_idx]
        if self.is_valid_coord(nxt_state):
            nxt_vals = self.qmap[nxt_state[0]][nxt_state[1]]
            nxt_max = max(nxt_vals)
        else:
            nxt_max = -1

        learnedval = (reward + self.disc * nxt_max )
        #print("reward:{0}, disc:{1}, nxtmax:{2}".format(reward, self.disc, nxt_max))
        qv = (1-self.lr) * oldval + self.lr * learnedval
        #print("lr:{0}, oldval:{1}, learnedval:{2}".format(self.lr, oldval, learnedval))
        return round(qv,4)

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
        """
        Determines whether a given coordinate is valid (in the map grid)
        :param coord: the coordinate to check
        :return: boolean True or False
        """
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
        """
        Checks whether all states have at least one action that is not zero
        """
        n_states_exclude = 2*self.w + 2*self.h -2 + 1
        return self.get_num_nonzero_states() >= self.w*self.h - n_states_exclude

    def get_optimal_path(self, startcoord, trophycoord):
        """
        Determines the optimal path per game episode given a starting state
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
        for i in range(len(coords)):
            coord = coords[i]
            try:
                path = self.get_optimal_path(coord, trophy_pos)
                #print("Optimal path for {0} is {2}".format(coord, path))
            except Exception as e:
                states.append(coord)
                print(e)
        return states

    def get_num_states_optimal_path(self, trophy_pos):
        """
        Gets the number of states with optimal paths to the trophy
        Note that scenarios where if by chance all equal max values result in shortest
        path are counted as optimal path.
        :return: an integer number of states
        """
        n_optimal=0
        coords=self.get_play_coords(trophy_pos)
        for i in range(len(coords)):
            coord = coords[i]
            try:
                path = self.get_optimal_path(coord, trophy_pos)
                if len(path)-1 == self._calc_shortest_dist(coord, trophy_pos):
                    #print("State with optimal path:{0}, dist:{1}".format(coord,len(path)-1))
                    n_optimal += 1
            except:
                pass
        return n_optimal

    def calc_current_score(self, trophy_pos):
        """
        Calculate the current training score as percent of states with optimal paths
        :return: the numeric score
        """
        n_total = self.get_num_play_coords(trophy_pos)
        n_optimal = self.get_num_states_optimal_path(trophy_pos)
        return n_optimal / n_total * 100

    def get_num_play_coords(self, trophy_pos):
        if self.n_play_coords == -1:
            self.n_play_coords = len(self.get_play_coords(trophy_pos))
        return self.n_play_coords

    def get_play_coords(self, trophy_pos, exclude_wall_states=True):
        """
        Gets all possible play coordinates for the snake.
        Excludes the trophy state and all wall coordinates
        :return: an array of coordinates
        """
        coords = []
        if exclude_wall_states:
            w1, w2, h1, h2 = 1, self.w-1, 1, self.h-1
        else:
            w1, w2, h1, h2 = 0, self.w, 0, self.h
        for i in range(h1,h2):
            for j in range(w1,w2):
                if not np.array_equal([j,i], trophy_pos):
                    coords.append([j,i])
        return coords

    def _calc_shortest_dist(self, coord, trophy_pos):
        d1=abs(trophy_pos[0]-coord[0])
        d2=abs(trophy_pos[1]-coord[1])
        return d1+d2            

