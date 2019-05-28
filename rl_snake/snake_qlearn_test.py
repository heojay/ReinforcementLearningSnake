# unit tests for snake_qlearn

import sys
import numpy as np
import snake_qlearn as env


def test_1():
        pass

def test_constr():
        app = env.SnakeQlearning()
        assert isinstance(app, env.SnakeQlearning)

def test_get_new_snake_start_coord():
        app = env.SnakeQlearning()
        app.play_initgame()
        s = app.snake
        coord = app.get_new_snake_start_coord()
        assert app.q.is_valid_coord(coord)
        assert not np.array_equal(s, coord)

def test_get_direction_to_coord():
        app = env.SnakeQlearning()
        assert app.get_direction_to_coord([0,0], [1,0]) == 1 # app.snake.RIGHT
        assert app.get_direction_to_coord([9,0], [8,0]) == 3 # app.snake.LEFT
        assert app.get_direction_to_coord([0,3], [0,4]) == 2 # app.snake.DOWN
        assert app.get_direction_to_coord([0,4], [0,3]) == 0 # app.snake.UP

def test_get_level_trophy():
        app = env.SnakeQlearning()
        app.TROPHY_POSITIONS = [[1,2], [3,4]]
        assert app.get_level_trophy(1) == [1,2]
        assert app.get_level_trophy(2) == [3,4]
        assert app.get_level_trophy(3) == [1,2]
        assert app.get_level_trophy(4) == [3,4]



## Test Qlearn
width=15 # set equal to tested class
height=15 # set equal to tested class
lr=0.5
disc=0.8

def test_Qlearn_constr():
        q = _get_a_Qlearn()
        assert isinstance(q, env.Qlearn)
        assert q.w == width
        assert q.h == height
        assert q.lr == lr
        assert q.disc == disc
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
        row=4
        col=5
        state=[col,row]
        nxt_state=[col,row+1]
        q = _get_a_Qlearn()
        q.qmap[col][row] = [0.2,0.8,0.5,0.0]
        action_idx=1
        reward=1
        qv = q.get_update_qvalue(reward, state, action_idx, nxt_state) # 1, state, 1, nxt
        assert qv == 0.9

def test_Qlearn_get_update_qvalue2():
        row=4
        col=5
        state=[col,row]
        nxt_state=[col,row+1]
        q = _get_a_Qlearn()
        q.qmap[col][row] = [0.2,0.8,0.5,0.0]
        q.qmap[col][row+1] = [0.5,0.5,0.5,0.5]
        action_idx=1
        reward=0
        qv = q.get_update_qvalue(reward, state, action_idx, nxt_state) # 1, state, 1, nxt
        assert qv == 0.6

def test_Qlearn_get_update_qvalue_offmap():
        row=0
        col=0
        state=[col,row]
        nxt_state=[col,row+1]
        q = _get_a_Qlearn()
        q.qmap[col][row] = [0.0,0.1,0.2,0.3]
        #q.qmap[col][row+1] = [0.5,0.5,0.5,0.5]
        action_idx=0
        reward=-1
        print(state)
        print(q.qmap[col][row])
        print(nxt_state)
        print(q.qmap[col-1][row])
        qv = q.get_update_qvalue(reward, state, action_idx, nxt_state) # 1, state, 1, nxt
        assert qv == -0.5
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

def test_Qlearn_get_play_coords_all():
    app = env.SnakeQlearning()
    app.play_initgame()
    exclude_wall_states = False
    states = app.q.get_play_coords(app.trophy_pos, exclude_wall_states)
    print(states)
    assert len(states) == (width*height) - 1

def test_Qlearn_get_play_coords_exclude_wall_states():
    app = env.SnakeQlearning()
    app.play_initgame()
    exclude_wall_states = True
    states = app.q.get_play_coords(app.trophy_pos, exclude_wall_states)
    print(states)
    assert len(states) == (width-2) * (height-2) - 1

def test_Qlearn_loop_play_states():
    app = env.SnakeQlearning()
    app.play_initgame()
    states = app.q.get_play_coords(app.trophy_pos)
    for i in range(len(states)):
        coord = states[i]
        print(coord)
    assert i == len(states)-1
    #assert False, "Fake assert to make PyTest output prints"

def test_Qlearn_calc_current_score_init():
    app = env.SnakeQlearning()
    app.play_initgame()
    assert app.q.calc_current_score(app.trophy_pos) >= 0
    assert app.q.calc_current_score(app.trophy_pos) < 100

def test_Qlearn_calc_current_score():
    app = env.SnakeQlearning()
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

def test_Qlearn_get_num_states_optimal_path():
    """
    This test will fail if the Q-table is not already trained.
    """
    app = env.SnakeQlearning()
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
    app = env.SnakeQlearning()
    app.play_initgame()
    assert app.q._calc_shortest_dist([0,0], [5,5]) == 10
    assert app.q._calc_shortest_dist([9,5], [4,8]) == 8


# private methods
def _get_a_Qlearn():
        return env.Qlearn(width, height, lr, disc)

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

