# unit tests for snake_qlearn2

import snake_qlearn2


def test_1():
    pass


def test_constr():
        app = snake_qlearn2.SnakeQlearning()
        assert isinstance(app, snake_qlearn2.SnakeQlearning)


"""
def test_play():
    app = snake_qlearn2.SnakeQlearning()
    app.play()
    print("Q-values")
    app.display_qvalues()
    #print("Optimal path")
    #path = app.q.get_optimal_path()
"""
