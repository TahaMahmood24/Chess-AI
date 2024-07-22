# Chess-AI

This project implements a Deep Q-Network (DQN) to play chess using the Stockfish engine as an opponent. It includes modules for managing the chess board, defining the DQN architecture, and training the DQN agent.

## Files

### 1. `chess_board.py`

This module provides a class `chess_board` for managing a chess game, including initializing the board, handling moves, encoding the board state, and integrating with the Stockfish chess engine.

#### Class: `chess_board`

**Methods:**
- `__init__(self, FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")`: Initializes the chess board with a given FEN string and sets up various attributes.
- `reset(self)`: Resets the board to the initial state.
- `print_board(self)`: Returns the current board state.
- `init_action_space(self)`: Initializes the action space for the chess board.
- `check_turn(self)`: Determines whose turn it is to move.
- `encode_fen(self)`: Encodes the FEN string into a layer board representation.
- `encode_legal_moves(self)`: Encodes the legal moves of the current board state.
- `make_random_move(self)`: Makes a random legal move on the board.
- `step(self, action)`: Executes a move on the board and returns the new state, reward, checkmate status, and the player who checkmated.
- `stockfish_move(self, elo_rating=10, depth=3)`: Uses the Stockfish engine to get the best move.
- `print_board_image(self, size=400, filename='board_image.png', simulate=False)`: Generates and optionally displays an SVG image of the board.
- `is_valid_fen(self)`: Checks if the current FEN string is valid.
- `decode_square(self, num)`: Decodes a square number into a chess notation square.
- `decode_move(self, tensor)`: Decodes a move from a tensor representation into a chess.Move object.

### 2. `chess_learn.py`

This module provides a class `Chess_Learn` for training a Deep Q-Network (DQN) agent to play chess using the Stockfish engine as an opponent.

#### Class: `Chess_Learn`

**Methods:**
- `__init__(self)`: Initializes the `Chess_Learn` class, sets up the device for computation, and initializes the Stockfish engine.
- `y(x, max_steps)`: A static method that calculates a value based on an exponential function.
- `match_outcome_counts(self, reward_list)`: Counts and returns the number of wins, losses, and ties from a list of rewards.
- `save_position_files(self, position_list, mode='train')`: Saves the positions of a chess game to a text file.
- `save_image_files(self, image, turn, step, mode='train')`: Saves the image of a chess board at a particular step to a file.
- `plot_progress(self, step_list, reward_list, new_elo_list)`: Plots the training progress, including steps, rewards, and Elo ratings.
- `calculate_elo(self, opponent_elo, reward_list)`: Calculates the new Elo rating based on the match outcomes against an opponent.
- `set_optimizer(self, policy_dqn)`: Sets the optimizer for the DQN model.
- `configure_and_get_random_top_move(self, opponent_elo, depth, FEN)`: Configures the Stockfish engine and gets the best move from its top recommendations.
- `save_model(self, policy_dqn, target_dqn, replay_memory, file_name='best_policy_dqn')`: Saves the model and replay memory to a file.
- `load_model(self, policy_dqn, target_dqn, replay_memory, file_name='best_policy_dqn')`: Loads the model and replay memory from a file.
- `learn(self, policy_dqn, target_dqn)`: The main training loop for the DQN agent.
- `test(self, policy_dqn, simulate=True, opponent_elo=1350, depth=10)`: Tests the DQN agent against the Stockfish engine.

### 3. `dqn.py`

This module provides the DQN class, which defines the architecture of a neural network used for Deep Q-Learning.

#### Class: `DQN`

**Attributes:**
- `fc1` (nn.Linear): First fully connected layer.
- `fc2` (nn.Linear): Second fully connected layer.
- `fc3` (nn.Linear): Third fully connected layer.
- `fc4` (nn.Linear): Fourth fully connected layer.
- `out` (nn.Linear): Output layer.

**Methods:**
- `__init__(self, in_states, h1_nodes, out_size)`: Initialize the DQN model with the given architecture.
- `forward(self, x)`: Perform a forward pass through the network.

## Usage Example

```python
# Importing the necessary classes
from chess_board import chess_board
from chess_learn import Chess_Learn
from dqn import DQN

# Initializing the chess board
board = chess_board()

# Initializing the DQN model
model = DQN(in_states=3, h1_nodes=512, out_size=8)

# Printing the initial board state
print(board.print_board())

# Making a random move
board.make_random_move()

# Printing the board state after the move
print(board.print_board())

# Initializing the learning class
chess_learner = Chess_Learn()

# Starting the learning process
chess_learner.learn(policy_dqn=model, target_dqn=model)

# Testing the model
chess_learner.test(policy_dqn=model)

