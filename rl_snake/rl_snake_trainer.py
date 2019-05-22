#!/usr/bin/env python3

# Trains the snake game using Q-learning
# The Q-learning hyperparameters are set in snake_qlearn.py

import sys
import time
import numpy as np
import snake_qlearn as env



if __name__ == "__main__":
    start = time.time()
    app = env.SnakeQlearning(fixed_snake=False, display=True)
    episodes = app.train()
    end = time.time()
    file_name = str.format("rl_gymsnake_{0}.log", time.strftime('%Y%m%d_%H%M%S'))
    #app.save_log(str.format("C:/Dev/logs/{0}", file_name))

    #app.display_qvalues()
    #qpath = 'C:/Dev/logs/rl_snake_qdata.txt'
    #app.save_qdata(qpath)

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

    app.plot_training_scores()
    print("Total train duration:{0} seconds" .format(round(end - start, 3)))

