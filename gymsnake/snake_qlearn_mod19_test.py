# unit tests for snake_qlearn_mod19

import sys
import numpy as np
import snake_qlearn_mod19



## Test Qlearn
width=15 # set equal to tested class
height=15 # set equal to tested class
lr=0.5
disc=0.8
expl=1.0
reward=0

def test_Qlearn_constr():
        q = _get_a_Qlearn()
        assert isinstance(q, snake_qlearn_mod19.Qlearn)
        assert q.w == width
        assert q.h == height
        assert q.lr == lr
        assert q.disc == disc
        assert q.expl == expl
        assert q.reward == reward
        assert q.qmap

def test_Qlearn_qmap():
        q = _get_a_Qlearn()
        numrows = len(q.qmap)
        assert numrows == height
        numcols = len(q.qmap[0])
        assert numcols == width
        for i in range(height):
                for j in range(width):
                        state = q.qmap[i][j]
                        print(i, j, state)
                        assert len(state) == 4
                        for a in range(len(state)):
                                assert state[a] == 0
        ##assert False, "Fake assert to make PyTest output prints"

def test_Qlearn_get_index_max_value():
        q = _get_modified_Qlearn() # values set at row=4,col=5
        assert q.get_index_max_value(5, 4) == 1

def test_Qlearn_get_update_qvalue():
        reward=1
        row=4
        col=5
        state=[col,row]
        action_idx=1
        nxt_state=[col,row+1]
        q = _get_a_Qlearn()
        state_vals = [0,0,0,0]
        state_vals[0] = 0.2
        state_vals[1] = 0.8
        state_vals[2] = 0.5
        state_vals[3] = 0.0
        q.qmap[col][row] = state_vals
        qv = q.get_update_qvalue(reward, state, action_idx, nxt_state) # 1, state, 1, nxt
        assert qv == 0.9
        ##assert False, "Fake assert to make PyTest output prints"

def test_Qlearn_get_optimal_path():
        q = _get_a_Qlearn()
        q.qmap[1][1]=[0,2,5,1]
        q.qmap[1][2]=[2,6,0,4]
        q.qmap[2][2]=[-1,0.2,0.8,0]
        startcoord=[1,1]
        trophycoord=[2,3]
        expected_path=[[1,1],[1,2],[2,2],[2,3]]
        path=q.get_optimal_path(startcoord, trophycoord)
        print("path len:" + str(len(path)))
        assert len(path) > 0
        assert len(path) < (width*height)
        assert len(path)==4
        assert path==expected_path
        #assert False, "Fake assert to make PyTest output prints"

def test_Qlearn_get_coord_next_max():
        q = _get_modified_Qlearn() # values set at row=4,col=5
        assert q.get_coord_next_max([5,4]) == [6,4]

def test_Qlearn_is_valid_coord():
        q = _get_a_Qlearn()
        assert q.is_valid_coord([0,0])
        assert q.is_valid_coord([5,9])
        assert q.is_valid_coord([3,-1]) == False
        assert q.is_valid_coord([25,3]) == False
        assert q.is_valid_coord([width,1]) == False
        assert q.is_valid_coord([9,height]) == False

def test_Qlearn_get_play_coords():
    app = snake_qlearn_mod19.SnakeQlearning()
    app.play_initgame()
    states = app.q.get_play_coords(app.trophy_pos)
    print(states)
    assert len(states) == (width*height) - (2*width+2*height-4) - 1
    #assert False, "Fake assert to make PyTest output prints"

def test_Qlearn_loop_play_states():
    app = snake_qlearn_mod19.SnakeQlearning()
    app.play_initgame()
    states = app.q.get_play_coords(app.trophy_pos)
    for i in range(len(states)):
        coord = states[i]
        print(coord)
    assert i == len(states)-1
    #assert False, "Fake assert to make PyTest output prints"

def test_Qlearn_calc_current_score_init():
    app = snake_qlearn_mod19.SnakeQlearning()
    app.play_initgame()
    assert app.q.calc_current_score(app.trophy_pos) >= 0
    assert app.q.calc_current_score(app.trophy_pos) < 100

def test_Qlearn_calc_current_score():
    app = snake_qlearn_mod19.SnakeQlearning()
    app.play_initgame()
    q = _get_a_Qlearn()
    q.qmap[1][1]=[0,2,5,1]
    q.qmap[1][2]=[2,6,0,4]
    q.qmap[2][2]=[-1,0.2,0.8,0]
    app.q = q
    expected_path=[[1,1],[1,2],[2,2],[2,3]]
    score = app.q.calc_current_score(app.trophy_pos) 
    print(score)
    assert score > 0
    assert score < 100

def test_get_num_states_optimal_path():
    app = snake_qlearn_mod19.SnakeQlearning()
    app.play_initgame()
    q = _get_a_Qlearn()
    q.qmap[1][1]=[0,2,5,1]
    q.qmap[1][2]=[2,6,0,4]
    q.qmap[2][2]=[-1,0.2,0.8,0]
    app.q = q
    test_coord = [1,1]
    expected_path=[[1,1],[1,2],[2,2],[2,3]]
    path = app.q.get_optimal_path(test_coord, app.trophy_pos)
    print(path)
    assert len(path)-1 == app.q._calc_shortest_dist(test_coord, app.trophy_pos)
    n_optimal = app.q.get_num_states_optimal_path(app.trophy_pos)
    assert n_optimal >= 1
    assert n_optimal < width*height

def test_Qlearn_calc_shortest_dist():
    app = snake_qlearn_mod19.SnakeQlearning()
    app.play_initgame()
    assert app.q._calc_shortest_dist([0,0], [5,5]) == 10
    assert app.q._calc_shortest_dist([9,5], [4,8]) == 8


# private methods
def _get_a_Qlearn():
        return snake_qlearn_mod19.Qlearn(width, height, lr, disc, expl, reward)

def _get_modified_Qlearn():
        q = _get_a_Qlearn()
        row = 4
        col = 5
        state = q.qmap[col][row]
        state[0] = 0.2
        state[1] = 0.8 # east
        state[2] = 0.5
        state[3] = 0.0
        q.qmap[col][row] = state
        return q

