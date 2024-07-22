"""
Script to train the DQN agent for chess using the Chess_Learn class.

This script initializes the Chess_Learn class and starts the training process for a specified number of matches. 

Usage:
    Run this script directly to start the training.

Classes:
    Chess_Learn: A class for training a Deep Q-Network (DQN) agent to play chess using the Stockfish engine as an opponent.

Functions:
    main(): Initializes the Chess_Learn class and starts the training process.

Example:
    python train.py
"""

from chess_learn import Chess_Learn

def main():
    """
    Initializes the Chess_Learn class and starts the training process for a specified number of matches.

    The training process involves playing a series of games against the Stockfish engine. The `simulate` parameter
    determines whether the training is run in simulation mode or not.
    """
    train_chess = Chess_Learn()
    train_chess.train(matches=10, simulate=False)

if __name__ == "__main__":
    main()