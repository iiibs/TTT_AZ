################
################
# requirements #
################
# numpy=1.24.6
# tensorflow=2.10.0

###########
# imports #
###########
import os
import json
import random
import math
import operator
from functools import reduce
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model,save_model

class Board:
 def __init__(self,cells=None):
  self._=[0]*9 if cells is None else list(cells)
  self.dimensions=(3,3)
  return
 def __len__(self):
  """Return the number of cells (always 9)."""
  return len(self._)
 def __getitem__(self, idx):
  """Get cell value(s) by index or slice."""
  return self._[idx]
 def __setitem__(self, idx, value):
  """Set cell value(s) by index or slice."""
  self._[idx] = value
 def __iter__(self):
  """Iterate over cell values."""
  return iter(self._)
 def __contains__(self, item):
  """Check if a value is present in the board."""
  return item in self._
 def __eq__(self, other):
  """Value-based equality with another Board or list-like."""
  return list(self) == list(other)
 def __repr__(self):
  """Debug representation."""
  return f"Board({self._!r})"
 def __array__(self, dtype=None):
  """Allow np.asarray(board) to return a NumPy array of cell values."""
  return np.array(self._, dtype=dtype)
 def to_list(self):
  """Return a shallow copy as a Python list."""
  return list(self._)
 @classmethod
 def from_list(cls, lst):
  """Create a Board from a flat list of 9 ints."""
  return cls(lst)
 def to_array(self):
  """Return a NumPy array copy of the board."""
  return np.array(self._)
 def get_available_moves(self):
  """Returns a list of empty-cell indices; robust to lists and arrays."""
  return [int(i) for i, v in enumerate(self._) if v == 0]
 def check_winner(self):
  """
  Returns 1 if player +1 wins, -1 if player -1 wins, 0 otherwise.
  Works with Python lists AND NumPy arrays of any shape/dtype.
  """
  # Normalize to flat 1D array so board[i] always returns a scalar
  #flattened_board=np.asarray(self).flatten()
  win_combos=[
   (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
   (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
   (0, 4, 8), (2, 4, 6),              # diagonals
  ]
  for c in win_combos:
   # Scalar chained comparison — works for int/float, list/array
   if self._[c[0]]==self._[c[1]]==self._[c[2]] != 0:
    return int(self._[c[0]])
  return 0
 def print_state(self):
  """Render the 3x3 board with X / O / . symbols and row/col labels."""
  sym = {1: "X", -1: "O", 0: "."}
  print("    0   1   2")
  print("  +---+---+---+")
  for row in range(3):
   cells = " | ".join(sym[self._[row * 3 + col]] for col in range(3))
   print(f"{row} | {cells} |")
   print("  +---+---+---+")
  print()
  return

 def is_terminal(self):
  ### True if game is won or the board is full.
  return self.check_winner() != 0 or len(self.get_available_moves()) == 0

 def apply_move(self,move,player):
  """
  Apply a move to the board and return the new board state.
  Accepts move as either a flat int (0-8) or (row, col) tuple.
  """
  # 1. Normalize board -> flat list of pure Python ints
  new_board_list=[int(x) for x in np.asarray(self._).flatten()]
  # 2. Compute flat index
  if isinstance(move,np.ndarray):
   if move.ndim != 1 or move.shape[0] != 2:
    raise ValueError(
     f"NumPy array move must be 1-D with 2 elements; got shape {move.shape}"
    )
   idx=int(move[0])*3+int(move[1])
  elif isinstance(move,(tuple,list)):
   r,c=move
   idx=int(r)*3+int(c)
  else: # int, np.int64, or any other scalar
   idx=int(move)
  # 3. Bounds check
  if idx<0 or idx>=9:
   raise IndexError(f"Move index {idx} is out of range for a 3x3 board (valid: 0-8).")
  # 4. Apply move (force Python int so board stays dtype-clean)
  new_board_list[idx]=int(player)
  return Board(new_board_list)

######################################################################x

DEFAULT_SETTINGS = {
 "n_repetitions":      20,
 "n_simulations":      20,
 "n_training_games":   20,
 "n_evaluation_games": 20,
 "b_detailed_train":False,
 "b_detailed_evaluate":False,
}

# Number of repeated generate-train-evaluate cycles #
n_repetitions=20
# Number os simulations in a game when creating a trining set - can be zero #
n_simulations=20
# Number of games when creating a training set - must be at least 1 #
n_training_games=20
# Number of games during evaluation of a neural network against random #
n_evaluation_games=20
b_detailed_train=False
b_detailed_evaluate=False

DATASET_FILE = 'position_tic_tac_toe_selfplay.pickle'
MODEL_FILE = 'convolutional_net_alphazero.keras'
SETTINGS_FILE='settings.json'

def save_settings(settings=None):
 """Write the current parameter values to disk as JSON."""
 if settings is None:
  settings = {
   "n_repetitions":n_repetitions,
   "n_simulations":n_simulations,
   "n_training_games":n_training_games,
   "n_evaluation_games":n_evaluation_games,
   "b_detailed_train":b_detailed_train,
   "b_detailed_evaluate":b_detailed_evaluate,
  }
 try:
  with open(SETTINGS_FILE, "w") as f:
   json.dump(settings, f, indent=4)
 except OSError as e:
  print(f"  ✗  Failed to save settings: {e}")
 return
def load_settings():
 """
 Load settings from disk at startup.
 If the file doesn't exist (or is corrupt), create it with defaults.
 """
 global n_repetitions,n_simulations,n_training_games,n_evaluation_games, \
        b_detailed_train,b_detailed_evaluate

 if os.path.exists(SETTINGS_FILE):
  try:
   with open(SETTINGS_FILE, "r") as f:
    settings = json.load(f)
  except (json.JSONDecodeError, OSError):
   print(f"  ⚠  Could not read '{SETTINGS_FILE}' — resetting to defaults.")
   settings = dict(DEFAULT_SETTINGS)
 else:
  settings = dict(DEFAULT_SETTINGS)
  save_settings(settings)
  print(f"  ℹ  No settings file found — created '{SETTINGS_FILE}' with defaults.")
 # Apply values to the globals, falling back to defaults for missing keys
 n_repetitions=settings.get("n_repetitions",DEFAULT_SETTINGS["n_repetitions"])
 n_simulations=settings.get("n_simulations",DEFAULT_SETTINGS["n_simulations"])
 n_training_games=settings.get("n_training_games",DEFAULT_SETTINGS["n_training_games"])
 n_evaluation_games=settings.get("n_evaluation_games",DEFAULT_SETTINGS["n_evaluation_games"])
 b_detailed_train=settings.get("b_detailed_train", DEFAULT_SETTINGS["b_detailed_train"])
 b_detailed_evaluate=settings.get("b_detailed_evaluate", DEFAULT_SETTINGS["b_detailed_evaluate"])
 return
def set_parameters():
 """
 Interactive menu to view and change the four global training parameters
 without editing the source code.
 Changes take effect immediately and persist for the entire session.
 Restart the program to reset values to the defaults above.
 """
 global n_repetitions,n_simulations,n_training_games,n_evaluation_games, \
        b_detailed_train,b_detailed_evaluate
 # Map menu key → (variable name, minimum allowed value, description)
 param_map = {
  "1": ("n_repetitions",1,"Generate-train-evaluate cycles   (>= 1)"),
  "2": ("n_simulations",0,"MCTS simulations per move        (>= 0; 0 = disabled)"),
  "3": ("n_training_games",1,"Self-play games per training set (>= 1)"),
  "4": ("n_evaluation_games",1,"Evaluation games vs random       (>= 1)"),
  "5": ("b_detailed_train",False,"Detailed logging of training"),
  "6": ("b_detailed_evaluate",False,"Detailed logging of evaluation"),
 }
 while True:
  # ── Display current values ──────────────────────────────────────────
  print("\n╔════════════════════════════════════════════╗")
  print(  "║            TTT_AZ — PARAMETERS             ║")
  print(  "╚════════════════════════════════════════════╝")
  print(f"  1. n_repetitions       = {n_repetitions:>6} (generate-train-evaluate cycles)")
  print(f"  2. n_simulations       = {n_simulations:>6} (MCTS simulations per move; 0 = disabled)")
  print(f"  3. n_training_games    = {n_training_games:>6} (self-play games per training set; >= 1)")
  print(f"  4. n_evaluation_games  = {n_evaluation_games:>6} (evaluation games vs random player)")
  print(f"  5. b_detailed_train    = {b_detailed_train:>6} (print details of training)")
  print(f"  6. b_detailed_evaluate = {b_detailed_evaluate:>6} (print details of evaluation)")
  print()
  choice = input("Enter parameter number to change, or 0 to return: ").strip()
  if choice == "0":
   print("  Settings saved for this session and to disk.")
   break
  if choice not in param_map:
   print(f"  ✗  '{choice}' is not a valid option — please enter 1, 2, 3, 4, or 0.")
   continue
  name, min_val, description = param_map[choice]
  current_val = globals()[name]
  if name.startswith("b_"):  # boolean parameter
   raw = input(f"  {description}\n"
               f"  Current value: {current_val}  →  New value (y/n): ").strip().lower()
   if raw in ("y", "yes", "true", "1"):
    new_val = True
   elif raw in ("n", "no", "false", "0"):
    new_val = False
   else:
    print(f"  ✗  '{raw}' is not a valid boolean. No change made.")
    continue
  else:
   raw = input(f"  {description}\n"
               f"  Current value: {current_val}  →  New value: ").strip()
   try:
    new_val = int(raw)
   except ValueError:
    print(f"  ✗  '{raw}' is not a valid integer. No change made.")
    continue
   if new_val < min_val:
    print(f"  ✗  Value must be >= {min_val}. No change made.")
    continue
  globals()[name] = new_val
  save_settings()   # ← persist immediately after every change
  print(f"  ✓  {name} updated: {current_val} → {new_val}")
 return
def mcts(board,player,neural_net,c_puct=1.0):
 """
 Monte Carlo Tree Search — AlphaZero-style.
 During expansion, the scalar prior probability for each move is now stored
 directly on the corresponding child node (child.P = float(policy_masked[move])).
 """
 global n_simulations
 # --- Fix: wrap raw list in Board so .is_terminal() etc. are available ---
 if isinstance(board, list):
  board = Board(board)
 # --- Node class (scoped inside mcts) ---
 class Node:
  def __init__(self,board,player):
   self.board    = board
   self.player   = player
   self.N        = 0
   self.W        = 0.0
   self.Q        = 0.0
   self.P        = None   # None until parent expansion assigns scalar prior
   self.children = {}     # move (int) -> Node
 tree = {}
 def run_simulation(root_board, root_player):
  path     = []
  node_key = (tuple(root_board), root_player)
  if node_key not in tree:
   tree[node_key] = Node(root_board, root_player)
  node = tree[node_key]
  path.append(node)
  # ── Selection ─────────────────────────────────────────────────────────
  while True:
   if node.board.is_terminal():
    break
   if not node.children:
    break
   total_N = sum(child.N for child in node.children.values())
   best_score, best_move = -float('inf'), None
   for move, child in node.children.items():
    # child.P is a scalar float assigned at expansion; never None here
    u     = c_puct * child.P * np.sqrt(total_N) / (1 + child.N)
    score = -child.Q + u
    if score > best_score:
     best_score, best_move = score, move
   node = node.children[best_move]
   path.append(node)
  # ── Expansion & Evaluation ─────────────────────────────────────────────
  value          = 0.0   # always overwritten in one of the two branches below
  node_board     = node.board
  current_player = node.player
  if node_board.is_terminal():
   winner = node_board.check_winner()
   if winner == 0:
    value = 0.0
   elif winner == current_player:
    value = 1.0
   else:
    value = -1.0
  else:
   available_moves = node_board.get_available_moves()
   canonical_board = [cell * current_player for cell in node_board._]
   board_array     = np.array(canonical_board, dtype=np.float32).reshape(1, 3, 3, 1)
   raw_policy, raw_value = neural_net.predict(board_array, verbose=0)
   policy = raw_policy.flatten()
   value  = float(raw_value[0][0])
   policy_masked = np.zeros(9, dtype=np.float32)
   for move in available_moves:
    policy_masked[move] = policy[move]
   if policy_masked.sum() > 0:
    policy_masked /= policy_masked.sum()
   else:
    policy_masked[available_moves] = 1.0 / len(available_moves)
   for move in available_moves:
    next_board = node_board.apply_move(move,current_player)
    child_key  = (tuple(next_board), -current_player)
    if child_key not in tree:
     tree[child_key] = Node(next_board, -current_player)
    node.children[move] = tree[child_key]
    # Store scalar prior on the child, not the parent ──────
    node.children[move].P = float(policy_masked[move])
  # ── Backpropagation ────────────────────────────────────────────────────
  for n in reversed(path):
   n.N += 1
   n.W += value
   n.Q  = n.W / n.N
   value = -value   # flip perspective at each level
 # Run n_simulations simulations
 root_key = (tuple(board._), player)
 if root_key not in tree:
  tree[root_key] = Node(board, player)
 root = tree[root_key]  
 # Expand the root explicitly
 root_board = root.board
 available_moves = root_board.get_available_moves()
 canonical_board = [cell * player for cell in root_board._]
 board_array = np.array(canonical_board, dtype=np.float32).reshape(1, 3, 3, 1)
 raw_policy, raw_value = neural_net.predict(board_array, verbose=0)
 policy = raw_policy.flatten()
 policy_masked = np.zeros(9, dtype=np.float32)
 for move in available_moves:
  policy_masked[move] = policy[move]
 if policy_masked.sum() > 0:
  policy_masked /= policy_masked.sum()
 else:
  policy_masked[available_moves] = 1.0 / len(available_moves)
 for move in available_moves:
  next_board=root_board.apply_move(move,player)
  child_key=(tuple(next_board),-player)
  if child_key not in tree:
   tree[child_key]=Node(next_board, -player)
  root.children[move]=tree[child_key]
  root.children[move].P=float(policy_masked[move])
 # Add Dirichlet noise
 legal_moves=available_moves
 alpha = 1.0          # see note below on choosing alpha
 epsilon = 0.25
 dirichlet_noise = np.random.dirichlet([alpha] * len(legal_moves))
 for i, move in enumerate(legal_moves):
  root.children[move].P = (1 - epsilon) * root.children[move].P + epsilon * dirichlet_noise[i]
 # Run mcts simulations
 for _ in range(n_simulations):
  run_simulation(board, player)
 # Build output policy from visit counts
 root         = tree[root_key]
 visit_counts = np.zeros(9, dtype=np.float32)
 for move, child in root.children.items():
  visit_counts[move] = child.N
 if visit_counts.sum() > 0:
  policy_out = visit_counts / visit_counts.sum()
 else:
  available  = board.get_available_moves()
  policy_out = np.zeros(9, dtype=np.float32)
  policy_out[available] = 1.0 / len(available)
 best_move = int(np.argmax(policy_out))
 return policy_out, best_move
def initialize_neural_network_and_training_dataset():
 """
 Create an AlphaZero-style neural network for Tic-Tac-Toe and and empty training dataset.

 The network takes a 3x3 board with a single channel as input and produces
 two outputs (dual-head architecture):
   - Policy head: probability distribution over 9 possible moves (softmax)
   - Value head:  scalar board evaluation in [-1, 1] (tanh)

 Fixes applied:
   1. Input shape uses tuple literal (board_dimensions, board_dimensions, 1)
      instead of the erroneous `board_dimensions + (1,)` (int + tuple TypeError).
   2. All layer references use tf.keras.layers (not a bare `layers` variable).
   3. Dual-head output: policy head (Dense 9, softmax) + value head (Dense 1, tanh).

 Returns:
     tf.keras.Model: Compiled AlphaZero-style model.
 """
 import os
 board=Board()
 board_dimensions = 3  # 3x3 Tic-Tac-Toe board
 # 1. Input layer
 input_layer = tf.keras.Input(
        shape=(board.dimensions[0], board.dimensions[1], 1),  # (3, 3, 1)
        dtype=tf.float32
 )
 # 2. Shared convolutional backbone
 x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(input_layer)
 x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
 x = tf.keras.layers.Flatten()(x)
 x = tf.keras.layers.Dense(128, activation='relu')(x)
 # 3. Dual-head output
 policy_head = tf.keras.layers.Dense(9, activation='softmax', name='policy')(x)
 value_head  = tf.keras.layers.Dense(1, activation='tanh',    name='value')(x)
 # create model
 model = tf.keras.Model(
        inputs=input_layer,
        outputs=[policy_head, value_head],
        name='AlphaZero_TicTacToe'
 )
 # compile model
 model.compile(
        optimizer='adam',
        loss={
            'policy': 'categorical_crossentropy',
            'value':  'mean_squared_error'
        },
        loss_weights={'policy': 1.0, 'value': 1.0}
 )
 # --- Save the model to disk for later loading ---
 model.save(MODEL_FILE)
 # --- Delete old training dataset (AlphaZero: fresh start per generation) ---
 if os.path.exists(DATASET_FILE):
  os.remove(DATASET_FILE)
  print(f"[create_neural_network] Deleted old dataset '{DATASET_FILE}'.")
 else:
  print(f"[create_neural_network] No existing dataset to delete.")
 return model
def cumulate_training_set(neural_net):
 """
 Play one self-play game using MCTS and the given neural network.
 Returns a list of (state, policy, value) tuples for training.
 """
 global n_repetitions,n_simulations,n_training_games,n_evaluation_games, \
        n_simulations
 game_history=[]
 board=Board()
 player=1
 # Play until terminal state
 while not board.is_terminal():
  # Canonicalize board for current player
  canonical_board=[x * player for x in board._]
  policy,move=mcts(canonical_board,player,neural_net=neural_net)
  # Store state and policy; value will be assigned after game ends
  game_history.append((canonical_board[:],policy.copy(),player))
  board=board.apply_move(move,player)
  winner=board.check_winner()
  if winner != 0 or board.is_terminal():
   break
  player=-player
 # Assign value to each tuple from the perspective of the player at that move
 finalized_history=[]
 for state,pol,p_at_move in game_history:
  if winner==0:
   value=0.0
  elif winner==p_at_move:
   value=1.0
  else:
   value=-1.0
  finalized_history.append((state,pol,value))
 return finalized_history
def train_supervised(game_spec,
                     network_file_path,
                     positions,
                     test_set_ratio=0.4,
                     regularization_coefficent=1e-5,
                     batch_size=100,
                     learn_rate=1e-4,
                     stop_turns_without_improvement = 7):
 """Train a network using supervised learning using against a list of game positions and moves chosen.
 We stop after we have had stop_turns_without_improvement without an improvement in the test error.
 The test set is used as a validation set as well, will possibly improve this in the future to have a seperate test
  and validation set.
 Args:
  stop_turns_without_improvement (int): we stop training after this many iterations without any improvement in
   the test error.
  regularization_coefficent (float): amount to multiply the l2 regularizer by in the loss function
  test_set_ratio (float): portion of the data to divide into the test set,
  positions ([(board_state, move)]): list of tuples of board states and the moves chosen in those board_states
  game_spec (games.base_game_spec.BaseGameSpec): The game we are playing
  create_network (->(input_layer : tf.placeholder, output_layer : tf.placeholder, variables : [tf.Variable])):
   Method that creates the network we will train.
  network_file_path (str): path to the file with weights we want to load for this network
  learn_rate (float):
  batch_size (int):
 Returns:
  episode_number, train_error, train_accuracy, new_test_error, test_accuracy
 """
 global b_detailed_train
 detailed=b_detailed_train
 # 1. Load existing model or create fresh as last resort
 if os.path.isfile(network_file_path):
  print(f"Loading model from '{network_file_path}'...")
  model = load_model(network_file_path)
 else:
  print(f"No saved model at '{network_file_path}'. Creating fresh model.")
  return None
 # Split the positions into training and test sets
 test_set_count = int(len(positions) * test_set_ratio)
 train_set = positions[:-test_set_count]
 test_set = positions[-test_set_count:]
 # Prepare the data
 train_inputs = [x[0] for x in train_set]
 train_outputs = [x[1] for x in train_set]
 test_inputs = [x[0] for x in test_set]
 test_outputs = [x[1] for x in test_set]
 # Compile the model
 model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learn_rate),
               loss=tf.keras.losses.MeanSquaredError(),
               metrics=['accuracy'])
 # Track the best test error
 best_test_error = float('inf')
 turns_without_test_improvement = 0
 episode_number = 1
 while True:
  # Shuffle the training set
  random.shuffle(train_set)
  # Train the model in batches
  train_error = 0
  for start_index in range(0, len(train_set) - batch_size + 1, batch_size):
   mini_batch_inputs = train_inputs[start_index:start_index + batch_size]
   mini_batch_outputs = train_outputs[start_index:start_index + batch_size]
   # Train on the mini-batch
   x_data = np.array(mini_batch_inputs).reshape(-1, 3,3,1)
   y_data = np.array(mini_batch_outputs)
   history = model.train_on_batch(x=x_data, y=y_data)
   train_error += history[0]  # history[0] is the loss
  # Reshape test inputs to match the model's input shape
  x_test = np.array([x[0] for x in test_set]).reshape(-1, 3, 3, 1)
  y_test = np.array([x[1] for x in test_set])
  # Evaluate the model on the test set
  results = model.evaluate(x=x_test, y=y_test, verbose=0)
  if detailed:
   print(model.metrics_names)  # See the order of returned values
   # Example: print all results with their names
   for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value}")
  # Access specific metrics by index if needed
  total_loss = results[0]
  policy_loss = results[1]
  value_loss = results[2]
  policy_accuracy = results[3]
  value_mae = results[4]
  test_error = total_loss
  test_accuracy = policy_accuracy
  if detailed:
   print("episode: %s train_error: %s test_error: %s test_acc: %s" %
         (episode_number, train_error, test_error, test_accuracy))
  # Check for improvement in test error
  if test_error < best_test_error:
   best_test_error = test_error
   turns_without_test_improvement = 0
  else:
   turns_without_test_improvement += 1
   if turns_without_test_improvement > stop_turns_without_improvement:
    if detailed:
     print("test error not improving for %s turns, ending training" % (stop_turns_without_improvement, ))
    break
  episode_number += 1
 # Save the trained model
 model.save(network_file_path)
 return episode_number, train_error, None, test_error, test_accuracy
def train_neural_network():
 #from techniques.train_supervised import train_supervised
 #from tic_tac_toe.network import tic_tac_toe_game_spec
 #import pickle
 print("Train policy and value neural network")
 board=Board()
 # Load pre-generated positions
 with open(DATASET_FILE, 'rb') as f:
  positions = pickle.load(f)
 # Train the neural network using supervised learning
 train_supervised(
  game_spec=board,
  network_file_path=MODEL_FILE,
  positions=positions,
  regularization_coefficent=1e-4
 )
 return
def human_choose_move(board):
 """
 Prompt the human for a move (0-8). Validates input and cell occupancy.
 Accepts RAW BOARD (list of 9 ints)
 """
 available_moves = board.get_available_moves()
 while True:
  try:
   move = int(input("Your move (0-8, row*3+col): ").strip())
  except ValueError:
   print("  Please enter an integer between 0 and 8.")
   continue
  if move not in range(9):
   print(f"  {move} is out of range. Choose from {available_moves}.")
  elif move not in available_moves:
   print(f"  Cell {move} is already occupied. Choose from {available_moves}.")
  else:
   return move
 return
def play_human_vs_nn(model):
 """
 Play one game between a human and the neural network.
 model      : trained Keras model (already loaded)
 """
 human_side=-1
 board  = Board()
 player = 1  # X always moves first
 print("\n" + "=" * 40)
 print(f"  You are {'X' if human_side == 1 else 'O'}  "
       f"({'first' if human_side == 1 else 'second'} player)")
 print("=" * 40)
 board.print_state()
 while not board.is_terminal():
  if player == human_side:
   print(f"\n--- Your turn ({'X' if player == 1 else 'O'}) ---")
   move = human_choose_move(board)
  else:
   print(f"\n--- AI thinking ({'X' if player == 1 else 'O'}) ---")
   # Canonical normalization: always from current player's perspective
   canonical_board = [x * player for x in board]
   policy, move = mcts(canonical_board, player, neural_net=model)
   print(f"    AI plays cell {move}")
  board = board.apply_move(move, player)
  board.print_state()
  winner = board.check_winner()
  if winner != 0:
   print("\n" + "=" * 40)
   print("  You WIN!" if winner == human_side else "  AI wins.")
   print("=" * 40 + "\n")
   return winner
  player = -player  # Alternate turns
 print("\n" + "=" * 40)
 print("  It's a DRAW!")
 print("=" * 40 + "\n")
 return 0
def human_test():
 print("Evaluate neural network against random")
 print(f"Loading model from: {MODEL_FILE}")
 model = tf.keras.models.load_model(MODEL_FILE)
 print(" Model loaded.\n")
 play_human_vs_nn(model)
 return
def alphazero_method():
 global n_repetitions,n_simulations,n_training_games,n_evaluation_games
 # Load model ONCE
 model=tf.keras.models.load_model(MODEL_FILE)
 # Load or initialize dataset
 try:
  with open(DATASET_FILE, 'rb') as f:
   training_data=pickle.load(f)
  print(f"Loaded {len(training_data)} positions from '{DATASET_FILE}'.")
 except FileNotFoundError:
  training_data=[]
  print("No existing dataset found. Starting fresh.")
 # Self-play: accumulate new data
 for _ in range(n_training_games):
  game_history=cumulate_training_set(neural_net=model)
  training_data.extend(game_history)
 # Save updated dataset
 with open(DATASET_FILE, 'wb') as f:
  pickle.dump(training_data, f)
 print(f"Saved {len(training_data)} positions to '{DATASET_FILE}'.")
 # Train the model
 train_neural_network()
 print(f"Model trained and saved to '{MODEL_FILE}'.")
 return

def main_menu():
 menu_labels={
  1:"Set parameters",
  2:"AlphaZero method",
  3:"Human test",
 }
 menu_actions={
  1:lambda:set_parameters(),
  2:lambda:alphazero_method(),
  3:lambda:human_test(),
 }
 while True:
  print("\nMain Menu:")
  print("----------")
  for i_menu in menu_labels:
   print(f"{i_menu}. {menu_labels[i_menu]}")
  choice=input("Enter your choice, or 0 to return: ")
  print()
  if choice=="0":
   break
  print()
  action=menu_actions.get(int(choice))
  action()
 return

if __name__=="__main__":
 load_settings()
 main_menu()
