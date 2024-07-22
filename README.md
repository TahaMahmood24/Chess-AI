# Chess-AI

Engineered a chess AI utilizing Deep Q-Learning Network (DQN) and Reinforcement Learning, autonomously learning optimal strategies through 1,000 gameplay simulations and achieving an 80 ELO rating performance

## Table of Contents

- [Introduction](#introduction)
- [Files](#files)
  - [train.py](#1-trainpy)
  - [test.py](#2-testpy)
  - [chess_board.py](#3-chess_boardpy)
  - [chess_learn.py](#4-chess_learnpy)
  - [dqn.py](#5-dqnpy)
  - [replaymemory.py](#6-replaymemorypy)
- [Deep Q-Network (DQN)](#deep-q-network-dqn)
- [Stockfish Integration](#stockfish-integration)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training and Testing](#training-and-testing)
- [Example Output](#example-output)
- [License](#license)

## Introduction

This project implements a Deep Q-Network (DQN) to play chess using the Stockfish engine as an opponent. It includes modules for managing the chess board, defining the DQN architecture, and training the DQN agent.

## Files

### 1. `train.py`

This is the main module to train a chess learning model.

### 2. `test.py`

This is the main module to test a chess learning model.

### 3. `chess_board.py`

This module provides a class `chess_board` for managing a chess game, including initializing the board, handling moves, encoding the board state, and integrating with the Stockfish chess engine.

### 4. `chess_learn.py`

This module provides a class `Chess_Learn` for training a Deep Q-Network (DQN) agent to play chess using the Stockfish engine as an opponent.

### 5. `dqn.py`

This module provides the DQN class, which defines the architecture of a neural network used for Deep Q-Learning.

### 6. `replaymemory.py`

This module provides the ReplayMemory class for storing and sampling transitions in a deque.

## Deep Q-Network (DQN)

Deep Q-Networks combine Q-Learning with deep neural networks. The agent learns to approximate the Q-value function, which estimates the expected reward for taking a certain action in a given state. The DQN is trained using experience replay, where past experiences are stored and sampled randomly for training, and target networks, which stabilize training by providing consistent targets for the Q-value updates.

## Stockfish Integration

Stockfish is a powerful open-source chess engine used as an opponent to train the DQN agent. It provides high-quality move suggestions and can be configured with different skill levels (Elo ratings) and search depths to adjust the difficulty of the training environment.

## Requirements

- Python 3.x
- PyTorch
- python-chess
- Stockfish

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TahaMahmood24/Chess-AI.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd Chess-AI
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Stockfish is installed and accessible in your PATH**:
   - Download Stockfish from [official website](https://stockfishchess.org/download/)
   - Extract the downloaded file and place the `stockfish` executable in a directory that's included in your system's PATH.

## Training and Testing

### Training

To train the DQN agent, run the training script:

```bash
python train.py --matches <number_of_matches> --simulate
```

#### Arguments:

- `--matches`: Number of matches to play (default: 2).
- `--simulate`: Run in simulation mode (default: False). Use this flag to enable simulation mode.

This script initializes the `Chess_Learn` class, sets up the DQN models, and starts the training loop. The progress, including rewards and Elo ratings, will be plotted and saved.

### Testing

To test the trained DQN agent, use the following script:

```bash
python test.py --matches <number_of_matches> --opponent_elo <opponent_elo> --depth <depth> --simulate
```

#### Arguments:

- `--matches`: Number of matches to play (default: 2).
- `--opponent_elo`: Elo rating of the opponent (default: 400).
- `--depth`: Search depth for the Stockfish engine (default: 3).
- `--simulate`: Run in simulation mode (default: False). Use this flag to enable simulation mode.

This script initializes the `Chess_Learn` class and starts the testing process for the specified number of matches, with the given opponent Elo rating, search depth, and simulation mode if enabled. The performance of the agent during testing will be reported based on the provided parameters.

## Example Output:

- For each match, plots will be generated and saved in the `test_data/images/match{match_number}` directory. Each image showcases the moves made during that match.
- After completing all the matches, a summary report will be generated, including:
  - The final Elo score, calculated using the following logic:

    ```python
    def calculate_elo(self, opponent_elo, reward_list, initial_elo=0, K=32):
        """
        Calculate the new Elo rating after a series of matches against a single opponent.

        :param initial_elo: Initial Elo rating of the player.
        :param opponent_elo: Elo rating of the opponent.
        :param reward_list: List of match outcomes (1 for win, 0 for tie, -1 for loss).
        :param K: K-factor in Elo rating formula, default is 32.
        :return: New Elo rating of the player.
        """
        current_elo = initial_elo

        for reward in reward_list:
            # Determine the actual score S
            if reward == 1:
                S = 1
            elif reward == 0:
                S = 0.5
            elif reward == -1:
                S = 0
            
            # Calculate the expected score E
            E = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
            
            # Update the current Elo rating
            current_elo += K * (S - E)

        return current_elo
    ```

  - The number of matches won, tied, and lost.
