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

import argparse
from chess_learn import Chess_Learn

def main(matches, opponent_elo, depth, simulate):
    """
    Initializes the Chess_Learn class and starts the testing process for a specified number of matches.

    Args:
        matches (int): Number of matches to play.
        opponent_elo (int): Elo rating of the opponent.
        depth (int): Search depth for the Stockfish engine.
        simulate (bool): Whether to run in simulation mode.
    """
    train_chess = Chess_Learn()
    train_chess.test(matches=matches, opponent_elo=opponent_elo, depth=depth, simulate=simulate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the DQN agent for chess using the Chess_Learn class.")
    parser.add_argument('--matches', type=int, default=2, help='Number of matches to play (default: 2)')
    parser.add_argument('--opponent_elo', type=int, default=400, help='Elo rating of the opponent (default: 400)')
    parser.add_argument('--depth', type=int, default=3, help='Search depth for the Stockfish engine (default: 3)')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode (default: False)')
    
    args = parser.parse_args()
    
    main(matches=args.matches, opponent_elo=args.opponent_elo, depth=args.depth, simulate=args.simulate)

