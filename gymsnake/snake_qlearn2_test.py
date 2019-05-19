# unit tests for snake_qlearn2

import sys
import snake_qlearn2


def test_1():
    pass


def test_constr():
        app = snake_qlearn2.SnakeQlearning()
        assert isinstance(app, snake_qlearn2.SnakeQlearning)

def test_possible_move_snake():
        # test if it is possible to move the snake
        app = snake_qlearn2.SnakeQlearning()
        app.play_initgame()
        app.snake.direction = app.snake.LEFT
        print(app.snake.direction)
        assert app.snake.direction == app.snake.LEFT
        # TODO also test possible to move snake
        print(app.snake.head)
        app.snake.head = [app.snake.head[0]+1, app.snake.head[1]]
        print(app.snake.head)
        ##assert False, "Fake assert to make PyTest output prints"

def test_metrics_dict():
        # to be used for per game level performance metrics
        d = dict(snakestart=[-1,-1], trophy=[-1,-1], disttotrophy=-1, beststepstrophy=-1, avgrewardsperstep=0)
        print(d)
        #assert False, "Fake assert to make PyTest output prints"


## Test Qlearn
width=10
height=10
lr=0.5
disc=0.8
expl=1.0
reward=0

def test_Qlearn_constr():
        q = _get_a_Qlearn()
        assert isinstance(q, snake_qlearn2.Qlearn)
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
        assert q.is_valid_coord([10,1]) == False
        assert q.is_valid_coord([9,10]) == False

def test_compare_list_equality():
    a = []
    b = []
    assert a == b
    c = [1]
    d = [1]
    assert c == d
    e = [101,9]
    f = [101,9]
    assert e == f
    g = [8,9]
    h = [7,4]
    assert g != h
    i = [1]
    j = [5,6]
    assert i != j


# private methods
def _get_a_Qlearn():
        return snake_qlearn2.Qlearn(width, height, lr, disc, expl, reward)

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



"""
def test_play():
    app = snake_qlearn2.SnakeQlearning()
    app.play()
    print("Q-values")
    app.display_qvalues()
    #print("Optimal path")
    #path = app.q.get_optimal_path()
"""
