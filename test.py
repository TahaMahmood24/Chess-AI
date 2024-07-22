"""
Script to test the DQN agent for chess using the Chess_Learn class.

This script initializes the Chess_Learn class and starts the testing process for a specified number of matches against the Stockfish engine with a given Elo rating and depth.

Usage:
    Run this script directly to start the testing.

Classes:
    Chess_Learn: A class for training and testing a Deep Q-Network (DQN) agent to play chess using the Stockfish engine as an opponent.

Functions:
    main(): Initializes the Chess_Learn class and starts the testing process.

Example:
    python test.py
"""

from chess_learn import Chess_Learn

def main():
    """
    Initializes the Chess_Learn class and starts the testing process for a specified number of matches.

    The testing process involves playing a series of games against the Stockfish engine. The parameters allow configuration of the number of matches, opponent's Elo rating, search depth, and simulation mode.
    """
    train_chess = Chess_Learn()
    train_chess.test(matches=2, opponent_elo=400, depth=3, simulate=False)

if __name__ == "__main__":
    main()
