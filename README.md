# Chess-AI

This project implements a Deep Q-Network (DQN) to play chess using the Stockfish engine as an opponent. It includes modules for managing the chess board, defining the DQN architecture, and training the DQN agent.

## Files

### 1. `main.py`

This is the main module is to train and test a chess learning model.

### 2. `chess_board.py`

This module provides a class `chess_board` for managing a chess game, including initializing the board, handling moves, encoding the board state, and integrating with the Stockfish chess engine.

### 3. `chess_learn.py`

This module provides a class `Chess_Learn` for training a Deep Q-Network (DQN) agent to play chess using the Stockfish engine as an opponent.

### 4. `dqn.py`

This module provides the DQN class, which defines the architecture of a neural network used for Deep Q-Learning.

### 5. `replaymemory.py`

This module provides the ReplayMemory class for storing and sampling transitions in a deque.
