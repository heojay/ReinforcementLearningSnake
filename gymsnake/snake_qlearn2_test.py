# unit tests for snake_qlearn2

import sys
import snake_qlearn2


def test_1():
    pass


def test_constr():
        app = snake_qlearn2.SnakeQlearning()
        assert isinstance(app, snake_qlearn2.SnakeQlearning)


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

def test_get_index_max_value():
        q = _get_modified_Qlearn() # values set at row=4,col=5
        assert q.get_index_max_value(5, 4) == 1

def test_get_update_qvalue():
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

def test_get_optimal_path():
        q = _get_a_Qlearn()
        q.qmap[1][1]=[0,]
        startcoord=[1,1]
        trophycoord=[3,3]
        expected_path=[[1,1],[1,2],[2,2],[2,3]]
        path=q.get_optimal_path(startcoord, trophycoord)
        print("path len:" + str(len(path)))
        assert len(path) > 0
        assert len(path) < (width*height)
        assert path==expected_path
        assert False, "Fake assert to make PyTest output prints"

def test_get_coord_next_max():
        q = _get_modified_Qlearn() # values set at row=4,col=5
        assert q.get_coord_next_max([5,4]) == [6,4]

def test_is_valid_coord():
        q = _get_a_Qlearn()
        assert q.is_valid_coord([0,0])
        assert q.is_valid_coord([5,9])
        assert q.is_valid_coord([3,-1]) == False
        assert q.is_valid_coord([25,3]) == False


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
