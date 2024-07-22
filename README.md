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
- [Usage Example](#usage-example)
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
python train.py
```

This script initializes the `Chess_Learn` class, sets up the DQN models, and starts the training loop. The progress, including rewards and Elo ratings, will be plotted and saved.

### Testing

To test the trained agent, run the testing script:

```bash
python test.py
```