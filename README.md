# ReinforcementLearningSnake

Research project for 1DL073 Natural Computation Methods for Machine Learning at Uppsala University.

### Supervisor

 * [Chencheng Liang](https://github.com/ChenchengLiang)
 
### Researchers
 
  * [Johan Alfredéen](https://github.com/terratrends)
  * [Jae Won Heo](https://github.com/yonseijaewon)
  * [Adam Ross](https://github.com/R055A)

## [Blind, OpenAI Gym](https://github.com/R055A/Group11/tree/master/rl_snake)

This implementation uses a modified version of the OpenAI gym from [https://github.com/grantsrb/Gym-Snake](https://github.com/grantsrb/Gym-Snake)

* Q-learning
* Epsilon decay

### Instructions

To train
    
    python3 rl_snake/rl_snake_trainer.py -t

To replay

    python3 rl_snake/rl_snake_trainer.py -r

## [Blind, custom](https://github.com/R055A/Group11/tree/master/reinforcement_learning)

Man cave developed Snake game and reinforcement learning environment, not based on anything prior aside from the concept of the classic Snake game. Trains and replays the following algorithms:

* Q-learning
* SARSA

### Instructions

To train Q-learning
    
    python3 rl_algorithms.py -q -l

To train SARSA

    python3 rl_algorithms.py -s -l

To playback Q-learning training

    python3 rl_algorithms.py -q -o
 
To playback SARSA training

    python3 rl_algorithms.py -s -o

To play the Snake game

    python3 rl_algorithms.py -m 

## [Semi-blind, PyGame Snake](https://github.com/R055A/Group11/tree/master/snake1)

* Q-learning
* Distance-based learning
