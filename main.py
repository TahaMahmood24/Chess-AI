"""
Main module to train and test a chess learning model using GPU.

This script utilizes the Chess_Learn class from the chess_learn module to train and test a chess AI model.

Classes:
    None

Functions:
    None

Usage Example:
    $ python main.py
"""

from chess_learn import Chess_Learn

if __name__ == "__main__":
    train_chess = Chess_Learn()
    train_chess.train(matches=10, simulate=False)
    train_chess.test(matches=2, opponent_elo=400, depth=3, simulate=False)
