# Code based on <Snake Game Python Tutorial> - freeCodeCamp.org
# https://www.youtube.com/watch?v=CD4qAhfFuLo&t=1926s
# https://pastebin.com/embed_js/jB6k06hG

import random
import pygame
import matplotlib.pyplot as plt
'''
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
'''

WIDTH = 500
ROWS = 20

DELAY = 5
FPS = 10

REWARD = {
    'snack': 1,
    'crash' : -1,
    'none' : -0.01,
    'start' : 0
}

ACTION = {
    0 : 'LEFT',
    1 : 'RIGHT',
    2 : 'UP',
    3 : 'DOWN'
}


class cube(object):
    def __init__(self, start, dirnx=1, dirny=0, color=(255, 0, 0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color

    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)

    # All cube has own direction
    # move to current direction

    def draw(self, surface, eyes=False):
        dis = WIDTH // ROWS
        i = self.pos[0]
        j = self.pos[1]

        pygame.draw.rect(surface, self.color, (i * dis + 1, j * dis + 1, dis - 2, dis - 2))

        if eyes:
            centre = dis // 2
            radius = 3
            circleMiddle = (i * dis + centre - radius, j * dis + 8)
            circleMiddle2 = (i * dis + dis - radius * 2, j * dis + 8)
            pygame.draw.circle(surface, (0, 0, 0), circleMiddle, radius)
            pygame.draw.circle(surface, (0, 0, 0), circleMiddle2, radius)


class snake(object):
    body = []
    turns = {}

    def __init__(self, color, pos):
        self.body = []
        self.turns = {}
        self.color = color
        self.head = cube(pos)
        self.body.append(self.head)
        self.dirnx = 1
        self.dirny = 0

    def move(self, action = None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            keys = pygame.key.get_pressed()

            for key in keys:
                if keys[pygame.K_LEFT] and self.dirnx != 1:
                    self.dirnx = -1
                    self.dirny = 0
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

                elif keys[pygame.K_RIGHT] and self.dirnx != -1:
                    self.dirnx = 1
                    self.dirny = 0
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

                elif keys[pygame.K_UP] and self.dirny != 1:
                    self.dirnx = 0
                    self.dirny = -1
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

                elif keys[pygame.K_DOWN] and self.dirny != -1:
                    self.dirnx = 0
                    self.dirny = 1
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        # if it's on the corner, it changes direction.

    def move_gym(self, action):

        key = ACTION[action]

        if key == 'LEFT' and self.dirnx != 1:
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        elif key == 'RIGHT' and self.dirnx != -1:
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        elif key == 'UP' and self.dirny != 1:
            self.dirnx = 0
            self.dirny = -1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        elif key == 'DOWN' and self.dirny != -1:
            self.dirnx = 0
            self.dirny = 1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        # if it's on the corner, it changes direction.

    def reset(self, pos):
        self.head = cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 1
        self.dirny = 0


    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(cube((tail.pos[0] - 1, tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(cube((tail.pos[0] + 1, tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(cube((tail.pos[0], tail.pos[1] - 1)))
        elif dx == 0 and dy == -1:
            self.body.append(cube((tail.pos[0], tail.pos[1] + 1)))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

def drawGrid(w, ROWS, surface):
    sizeBtwn = w // ROWS

    x = 0
    y = 0
    for l in range(ROWS):
        x = x + sizeBtwn
        y = y + sizeBtwn

        pygame.draw.line(surface, (255, 255, 255), (x, 0), (x, w))
        pygame.draw.line(surface, (255, 255, 255), (0, y), (w, y))


def redrawWindow(surface):
    global ROWS, WIDTH, s, snack
    surface.fill((0, 0, 0))
    s.draw(surface)
    snack.draw(surface)
    drawGrid(WIDTH, ROWS, surface)
    pygame.display.update()


def randomSnack(ROWS, item):
    positions = item.body

    while True:
        x = random.randrange(ROWS)
        y = random.randrange(ROWS)
        if len(list(filter(lambda z: z.pos == (x, y), positions))) > 0:
            continue
        else:
            break

    return (x, y)


class Env:
    def __init__(self):
        self.state_size = 4 #state
        self.action_size = 4

    def get_state(self):
        head_pos = self.s.head.pos
        snack_pos = self.snack.pos

        return head_pos[0] - snack_pos[0], head_pos[1] - snack_pos[1]

    def reset(self):
        self.s = snake((255, 0, 0), (10, 10))
        self.snack = cube(randomSnack(ROWS, self.s), color=(0, 255, 0))

        state = self.get_state() #Can be anything that you want.
        return state, 0, False, 'start'

    def step(self, action):
        self.s.move_gym(action)
        info = 'none'
        done = False
        head_pos = self.s.head.pos

        if self.s.head.pos == self.snack.pos:
            self.s.addCube()
            self.snack = cube(randomSnack(ROWS, self.s), color=(0, 255, 0))
            info = 'snack'

        else:

            if head_pos[0] >= ROWS or head_pos[0] < 0 or head_pos[1] >= ROWS or head_pos[1] < 0:
                info = 'crash'
                done = True

            else:
                for x in range(len(self.s.body)):
                    if self.s.body[x].pos in list(map(lambda z: z.pos, self.s.body[x + 1:])):
                        info = 'crash'
                        done = True

        snack_pos = self.snack.pos
        reward = REWARD[info]

        state = self.get_state()

        return state, reward, done, info


def human():
    global s, snack
    win = pygame.display.set_mode((WIDTH, WIDTH))
    s = snake((255, 0, 0), (10, 10))
    snack = cube(randomSnack(ROWS, s), color=(0, 255, 0))
    flag = True

    clock = pygame.time.Clock()

    while flag:
        pygame.time.delay(DELAY)
        clock.tick(FPS)
        s.move_gym(shortest_pass(s, snack))
        if s.body[0].pos == snack.pos:
            s.addCube()
            snack = cube(randomSnack(ROWS, s), color=(0, 255, 0))

        head = s.body[0].pos

        if head[0] >= ROWS or head[0] < 0 or head[1] >= ROWS or head[1] < 0:
            print("Score: ", len(s.body) - 1)
            flag = False
            break

        for x in range(len(s.body)):
            if s.body[x].pos in list(map(lambda z: z.pos, s.body[x + 1:])):
                print("Score: ", len(s.body) - 1)
                flag = False
                break

        redrawWindow(win)

    pass


def shortest_pass(s, t):
    head = s.head.pos
    snack = t.pos

    dirnx, dirny = s.dirnx, s.dirny
    x_gap = snack[0] - head[0] #LEFT RIGHT
    y_gap = snack[1] - head[1] #UP DOWN

    if(x_gap == 0):
        if(y_gap > 0):
            return 3
        else:
            return 2

    elif(y_gap == 0):
        if(x_gap > 0):
            return 1
        else:
            return 0

    else:
        if(dirnx * x_gap < 0):
            if(y_gap > 0):
                return 3
            else:
                return 2
        else:
            if(x_gap > 0):
                return 1
            else:
                return 0




class DQNAgent:
    def __init__(self, state_size, action_size):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)




env = Env()
state_size = 2
action_size = 4

'''

agent = DQNAgent(state_size, action_size)

episodes = 1000

scores = []

for e in range(episodes):
    done = False
    s = 0
    state, reward, done, info = env.reset()
    state = np.reshape(state, [1,state_size])

    while not done:

        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1,state_size])
        next_action = agent.get_action(next_state)
        agent.append_sample(state, action, reward, next_state, done)

        if len(agent.memory) >= agent.train_start:
            agent.train_model()

        s += reward
        state = next_state

        if done:
            agent.update_target_model()
            scores.append(s)
    print(s)

plt.plot(scores)
plt.show()


observation, reward, done, info = env.reset()

for i in range(100):

    action = random.randint(0,3)
    print(observation, action, reward, info)
    next_observation, reward, done, info = env.step(action)
    observation = next_observation
    if done:
       observation, reward, done, info = env.reset()
'''

human()