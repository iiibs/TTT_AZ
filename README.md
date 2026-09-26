# TTT_AZ
Train neural network to play tic-tac-toe using the AlphaZero method

## Run

Load the TTT_AZ.sln solution in Visual Studio and run.

## Reference

As a basis, I used this repo of Daniel Slater:
 https://github.com/DanielSlater/AlphaToe/

Differences from that repo:
 - I am dealing with tic-tac-toe only, and only the case when the neural network starts the game.
   So I was able to decrease the number of files, and the length of code to a single source file with 700 lines.
 - I do not use a database of games against any more advanced player.
   For training just games against random are used, combined with the MCTS method.
 - I do not use minimax, or any programmed tic-tac-toe knowledge or rules at all.
 - I created a simple menu for the settings, the AlphaZero method, and for human testing.
   A human can decide after not more than five games against the neural network how efficient it is.

## Main menu

 Main Menu:
 ----------
 1. Set parameters
 2. Alphazero method
 3. Human test

TTT_AZ - PARAMETERS
-------------------
 1. n_repetitions       =      2 (generate-train-evaluate cycles)
 2. n_simulations       =      2 (MCTS simulations per move; 0 = disabled)
 3. n_training_games    =      2 (self-play games per training set; >= 1)
 4. n_evaluation_games  =      2 (evaluation games vs random player)
 5. b_detailed_train    =      1 (print details of training)
 6. b_detailed_evaluate =      1 (print details of evaluation)
