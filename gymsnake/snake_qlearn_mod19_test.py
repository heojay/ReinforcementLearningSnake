# unit tests for snake_qlearn_mod19

import sys
import numpy as np
import snake_qlearn_mod19



## Test Qlearn
width=10
height=10
lr=0.5
disc=0.8
expl=1.0
reward=0

def test_Qlearn_get_play_coords():
    app = snake_qlearn_mod19.SnakeQlearning()
    app.play_initgame()
    states = app.q.get_play_coords(app.trophy_pos)
    print(states)
    assert len(states) == 100-1-36
    #assert False, "Fake assert to make PyTest output prints"

def test_Qlearn_loop_play_states():
    app = snake_qlearn_mod19.SnakeQlearning()
    app.play_initgame()
    states = app.q.get_play_coords(app.trophy_pos)
    for i in range(len(states)):
        coord = states[i]
        print(coord)
    assert i == len(states)-1
    assert False, "Fake assert to make PyTest output prints"


