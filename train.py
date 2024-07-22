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
import argparse

def main(matches, simulate):
    """
    Initializes the Chess_Learn class and starts the training process for a specified number of matches.

    Args:
        matches (int): Number of matches to play during training.
        simulate (bool): Whether to run in simulation mode.
    """
    train_chess = Chess_Learn()
    train_chess.train(matches=matches, simulate=simulate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DQN agent for chess using the Chess_Learn class.")
    parser.add_argument('--matches', type=int, default=10, help='Number of matches to play during training (default: 10)')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode (default: False)')
    
    args = parser.parse_args()
    
    main(matches=args.matches, simulate=args.simulate)