"""
chess_learn.py

This module provides a class `Chess_Learn` for training a Deep Q-Network (DQN) agent to play chess using the Stockfish engine 
as an opponent. The module includes various hyperparameters, methods for training, saving, and testing the DQN model, 
and utilities for handling chess matches and calculating Elo ratings.

Classes:
    Chess_Learn: A class to handle the learning process for a chess-playing DQN agent.

Methods:
    __init__(self):
        Initializes the Chess_Learn class, sets up the device for computation, and initializes the Stockfish engine.

    y(x, max_steps):
        A static method that calculates a value based on an exponential function.

    match_outcome_counts(self, reward_list):
        Counts and returns the number of wins, losses, and ties from a list of rewards.

    save_position_files(self, position_list, mode='train'):
        Saves the positions of a chess game to a text file.

    save_image_files(self, image, turn, step, mode='train'):
        Saves the image of a chess board at a particular step to a file.

    plot_progress(self, step_list, reward_list, new_elo_list):
        Plots the training progress, including steps, rewards, and Elo ratings.

    calculate_elo(self, opponent_elo, reward_list):
        Calculates the new Elo rating based on the match outcomes against an opponent.

    set_optimizer(self, policy_dqn):
        Sets the optimizer for the DQN model.

    configure_and_get_random_top_move(self, opponent_elo, depth, FEN):
        Configures the Stockfish engine and gets the best move from its top recommendations.

    save_model(self, policy_dqn, target_dqn, replay_memory, file_name='best_policy_dqn'):
        Saves the model and replay memory to a file.

    load_model(self, policy_dqn, target_dqn, replay_memory, file_name='best_policy_dqn'):
        Loads the model and replay memory from a file.

    learn(self, policy_dqn, target_dqn):
        The main training loop for the DQN agent.

    test(self, policy_dqn, simulate=True, opponent_elo=1350, depth=10):
        Tests the DQN agent against the Stockfish engine.
"""

import chess
import chess.svg
import numpy as np
import random
import torch
import shutil
from torch import nn
from IPython.display import display, SVG
from stockfish import Stockfish
import glob
import os
import re

from chess_board import chess_board
from dqn import DQN
from replaymemory import ReplayMemory


class Chess_Learn:
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.99        # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 64            # size of the training data set sampled from the replay memory
    step_level = 5
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.
    max_steps = 100
    part = 2
    move_part = 1/part

    stockfish_engine = Stockfish(path=r"stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")

    def __init__(self):
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    @staticmethod
    def y(x, max_steps):
        return min(1, (1 / (np.e - 1)) * (np.exp(x / max_steps) - 1))
    
    def match_outcome_counts(self, reward_list):
    # Initialize counters
        wins = 0
        losses = 0
        ties = 0

        # Count the number of wins, losses, and ties
        for reward in reward_list:
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            elif reward == 0:
                ties += 1

        # Return the results as a dictionary
        return {
            "Total matches": len(reward_list),
            "Matches won": wins,
            "Matches lost": losses,
            "Matches tied": ties
        }
    
    def load_and_simulate_match(self, mode='test',match=1):
        # Load the list of FEN positions from the .npy file
        filepath = f'{mode}_data/positions/positions{match}.npy'
        fen_positions = np.load(filepath, allow_pickle=True).tolist()
        
        # Simulate the match by showing each position
        for idx, fen in enumerate(fen_positions):
            print(f"Position {idx + 1}: {fen}")
            
            # Create a chess board from the FEN string
            board = chess.Board(fen)
            
            # Render the board as an SVG image
            svg = chess.svg.board(board=board, size=400)
            
            # Display the SVG image using IPython.display
            display(SVG(svg))
    
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
    
    def configure_and_get_best_move(self, skill_level, depth, stockfish_engine, state_fen):
        """
        Configure the Stockfish engine and get the best move.

        :param stockfish_engine: The Stockfish engine instance.
        :param state_fen: The FEN position of the current state.
        :param skill_level: The skill level to set for the Stockfish engine, default is 2600.
        :param depth: The search depth for the Stockfish engine, default is 10.
        :return: The best move suggested by the Stockfish engine.
        """
        stockfish_engine.set_skill_level(skill_level)
        stockfish_engine.set_depth(depth)
        stockfish_engine.set_fen_position(state_fen)
        return stockfish_engine.get_best_move()
    
    def configure_and_get_random_top_move(self, skill_level, depth, state_fen, top_moves_count=7):
        """
        Configure the Stockfish engine and get a random top move.

        :param skill_level: The skill level to set for the Stockfish engine.
        :param depth: The search depth for the Stockfish engine.
        :param state_fen: The FEN position of the current state.
        :param top_moves_count: The number of top moves to consider, default is 7.
        :return: A randomly selected move from the top moves.
        """
        self.stockfish_engine.set_skill_level(skill_level)
        self.stockfish_engine.set_depth(depth)
        self.stockfish_engine.set_fen_position(state_fen)
        best_moves = self.stockfish_engine.get_top_moves(top_moves_count)
        selected_move = random.choice(best_moves)['Move']
        return chess.Move.from_uci(selected_move)

    def delete_files(self, data_type='train'):
        folder_paths = [
            f"{data_type}_data/positions/*.npy"
        ]
        
        # Iterate over each folder path and delete all .npy files
        for path in folder_paths:
            files = glob.glob(path)
            for file in files:
                os.remove(file)
                print(f"Deleted: {file}")

    def delete_folders(self, data_type='train'):
        # Define the path to the directory containing subdirectories
        parent_directory = f"{data_type}_data/images"
        
        # Check if the parent directory exists
        if os.path.isdir(parent_directory):
            # Iterate over all items in the parent directory
            for item in os.listdir(parent_directory):
                item_path = os.path.join(parent_directory, item)
                # Check if the item is a directory
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted subdirectory: {item_path}")

    def findNextIdx(self, mode):
        files = glob.glob(f"{mode}_data/positions/*.npy")
        if len(files) == 0:
            return 1  # if no files, return 1
        highestIdx = 0
        for f in files:
            match = re.search(r"positions(\d+)/.npy", f)
            if match:
                currIdx = int(match.group(1))
                highestIdx = max(highestIdx, currIdx)
        return highestIdx + 1


    def save_position_files(self, fen_list, mode='train'):
        nextIdx = self.findNextIdx(mode)
        np.save(f"{mode}_data/positions/positions{nextIdx}.npy", np.array(fen_list))
        # print(f"Match #{nextIdx}: Saved successfully")

        return None
    
    def get_piece_color(self, piece):
        return 'White' if piece else 'Black'

    def save_image_files(self, svg_data, match, piece, move, mode='train'):
        nextIdx = self.findNextIdx(mode)
        
        # Create directory if it does not exist
        directory = f"{mode}_data/images/match{match}"
        os.makedirs(directory, exist_ok=True)
        piece_color = self.get_piece_color(piece)
        
        # Define the SVG file path
        svg_path = f"{directory}/image{move}_{piece_color}.svg"
        
        # Ensure svg_data is bytes
        if isinstance(svg_data, str):
            svg_data = svg_data.encode('utf-8')
        
        # Write SVG data directly to the file
        with open(svg_path, 'wb') as f:
            f.write(svg_data)
        
        # print(f"Match #{nextIdx}: Saved SVG image as {svg_path}")
        return None

    def train(self, matches, simulate):
        state = chess_board()
        self.delete_files('train')
        self.delete_folders('train')
        num_states = state.layer_board.flatten().shape[0]
        num_actions = state.one_hot_tensor.flatten().shape[0]
        print(f"Num_States: {num_states}, Num_Actions: {num_actions}")

        num_channels = 12
        h1_nodes = 256  # Adjust this as needed
        output_size = 64  # Since output shape is [64, 64]

        policy_dqn = DQN(in_states=num_channels, h1_nodes=h1_nodes, out_size=output_size).to(self.device)
        target_dqn = DQN(in_states=num_channels, h1_nodes=h1_nodes, out_size=output_size).to(self.device)

        epsilon = 1  # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_match = np.zeros(matches)

        # List to keep track of epsilon decay
        epsilon_history = []
        reward_list = []
        step_list = []
        position_list = []
        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0      

        for match in range(matches):
            checkmate = False      
            win = ''
            step_count = 0
            state = chess_board()
            #print(f"Match: {match}")

            while not checkmate and step_count < self.max_steps:
                turn = state.check_turn()
                move_part = self.y(step_count, self.step_level)
                    
                if turn[0]:
                    # Select action based on epsilon-greedy
                    if random.random() < epsilon:
                        # select random action
                        if random.random() > move_part:
                            action = state.make_random_move()
                        else:

                            action = self.configure_and_get_best_move(2600, 10, self.stockfish_engine, state.FEN)
                    else:
                        # select best action            
                        with torch.no_grad():
                            self.stockfish_engine.set_fen_position(state.FEN)
                            action = self.stockfish_engine.get_best_move()

                else:
                    action = self.configure_and_get_best_move(400, 3, self.stockfish_engine, state.FEN)
                    

                # Execute action
                new_state, reward, checkmate, win = state.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, checkmate, win))

                if checkmate:
                    reward_list.append(reward)
                    step_list.append(step_count)
                    self.save_position_files(position_list, 'train')
                    self.image = state.print_board_image(simulate=simulate)
                    self.save_image_files(self.image, match+1, turn[0], step_count, mode='train')
                    print(f"Checkmate has occurred after {step_count} moves. Winner is {win}")
            
                position_list.append(state.FEN)
                self.image = state.print_board_image(simulate=simulate)
                self.save_image_files(self.image, match+1, turn[0], step_count, mode='train')
                state = new_state   
                step_count += 1

                if step_count == self.max_steps:
                    print(f"Max Steps of {self.max_steps} reached")
                    self.save_position_files(position_list, 'train')
                    self.image = state.print_board_image(simulate=simulate)
                    self.save_image_files(self.image, match+1, turn[0], step_count, mode='train')
                    reward_list.append(reward)                

            # Train the network if enough experience has been collected
            if reward_list.count(1) > 0:
                self.mini_batch_size = len(memory)
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/matches, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        torch.save(policy_dqn.state_dict(), "chess_model.pt")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, checkmate, win in mini_batch:
            if checkmate:
                target = torch.tensor([reward], dtype=torch.float32, device=self.device)
            else:
                with torch.no_grad():
                    target = torch.tensor(
                        reward + self.discount_factor_g * target_dqn(new_state.encode_fen().unsqueeze(0).to(self.device)).max()
                    )

            current_q = policy_dqn(state.encode_fen().unsqueeze(0).to(self.device))
            current_q_list.append(current_q)

            target_q = target_dqn(state.encode_fen().unsqueeze(0).to(self.device))
            action_tensor, start_square, end_square = state.encode_move(action)
            target_q[start_square, end_square] = target.item()
            target_q_list.append(target_q)

        current_q_list = torch.cat(current_q_list)
        target_q_list = torch.cat(target_q_list)

        loss = self.loss_fn(current_q_list, target_q_list)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        torch.save(policy_dqn.state_dict(), "chess_model.pt")

    def test(self, matches, opponent_elo=400, depth=3, simulate=False):
        num_channels = 12
        h1_nodes = 256  # Adjust this as needed
        output_size = 64  # Since output shape is [64, 64]
        self.delete_files('test')
        self.delete_folders('test')
        policy_dqn = DQN(in_states=num_channels, h1_nodes=h1_nodes, out_size=output_size).to(self.device)
        policy_dqn.load_state_dict(torch.load("chess_model.pt"))
        policy_dqn.eval()

        reward_list = []
        step_list = []

        for match in range(matches):
            state = chess_board()
            checkmate = False
            step_count = 0
            position_list = []
            print(f"Match #{match+1}:") if simulate else None

            while not checkmate and step_count < self.max_steps:
                turn = state.check_turn()

                if turn[0]:
                    action = policy_dqn(state.encode_fen().unsqueeze(0).to(self.device))
                    action = state.decode_move(action)

                else:
                    action = self.configure_and_get_random_top_move(opponent_elo, depth, state.FEN)
                    

                new_state, reward, checkmate, win = state.step(action)

                position_list.append(state.FEN)
                self.image = state.print_board_image(simulate=simulate)
                self.save_image_files(self.image, match+1, turn[0], step_count, mode='test')
                state = new_state   
                step_count += 1

                if checkmate:
                    reward_list.append(reward)
                    step_list.append(step_count)
                    position_list.append(state.FEN)
                    self.image = state.print_board_image(simulate=simulate)
                    self.save_image_files(self.image, match+1, turn[0], step_count, mode='test')
                    self.save_position_files(position_list, 'test')
                    print(f"Checkmate has occurred after {step_count} moves. Winner is {win}")

                if step_count == self.max_steps:
                    reward_list.append(reward)
                    step_list.append(step_count)
                    position_list.append(state.FEN)
                    self.image = state.print_board_image(simulate=simulate)
                    self.save_image_files(self.image, match+1, turn[0], step_count, mode='test')
                    self.save_position_files(position_list, 'test')
                    print(f"Max Steps of {self.max_steps} reached")

        outcome_counts = self.match_outcome_counts(reward_list)
        new_elo = self.calculate_elo(opponent_elo, reward_list)
        print(outcome_counts)
        print(f"New Elo rating: {new_elo}")
