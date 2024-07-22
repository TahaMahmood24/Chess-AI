import chess
import chess.svg
import numpy as np
import random
import torch
from IPython.display import display, SVG

class chess_board:
    def __init__(self, FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        self.FEN = FEN
        self.board = chess.Board(self.FEN)
        self.init_action_space()
        self.layer_board = torch.zeros((12, 8, 8), dtype=torch.float32)
        self.encode_fen()
        self.uci_legal_moves = None
        self.san_legal_moves = None
        self.encoded_legal_moves = None
        self.checkmate = False
        self.reward = 0
        self.stockfish = None  # Initialize Stockfish if needed
        self.one_hot_tensor = torch.zeros(size=(64, 64), dtype=torch.int)

    def reset(self):
        self.FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.board.set_fen(self.FEN)
        self.encode_fen()

    def print_board(self):
        return self.board

    def init_action_space(self):
        self.action_space = np.zeros((64, 64))

    def check_turn(self):
        """Make a random legal move on the board."""
        if self.FEN.split(' ')[-5] == 'w':
            self.turn = True, 'White'
        else:
            self.turn = False, 'Black'

        return self.turn    

    def encode_fen(self):
        pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        piece_to_index = {piece: idx for idx, piece in enumerate(pieces)}

        rows = self.FEN.split(' ')[0].split('/')
        for i, row in enumerate(rows):
            col = 0
            for char in row:
                if char.isdigit():
                    col += int(char)
                else:
                    index = piece_to_index[char]
                    self.layer_board[index, i, col] = 1 if char.isupper() else -1
                    col += 1

        self.layer_board = self.layer_board.clone().detach()
        return self.layer_board

    def get_legal_moves(self):
        legal_moves = list(self.board.legal_moves)
        self.uci_legal_moves = [move.uci() for move in legal_moves]

    def get_legal_moves_san(self):
        legal_moves = list(self.board.legal_moves)
        self.san_legal_moves = [self.board.san(move) for move in legal_moves]

    def encode_square(self, square):
        file = ord(square[0]) - ord('a')
        rank = int(square[1]) - 1
        return file + 8 * rank

    def encode_move(self, move):
        if isinstance(move, chess.Move):
            move = move.uci()
        
        start_square = move[:2]
        end_square = move[2:4]
        encoded_start = self.encode_square(start_square)
        encoded_end = self.encode_square(end_square)
        one_hot_tensor = torch.zeros((64, 64), dtype=torch.int)
        one_hot_tensor[encoded_start, encoded_end] = 1
        return one_hot_tensor, encoded_start, encoded_end

    def encode_legal_moves(self):
        empty_list = []
        if self.san_legal_moves is None or self.uci_legal_moves is None:
            raise ValueError("Legal moves not initialized. Call get_legal_moves() and get_legal_moves_san() first.")

        piece_to_num = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
        for san_move, uci_move in zip(self.san_legal_moves, self.uci_legal_moves):
            if san_move in ["O-O", "O-O-O"]:
                empty_list.append([64 if san_move == "O-O" else 65, 0, 0])
            else:
                start_square = uci_move[:2]
                end_square = uci_move[2:4]
                piece_num = piece_to_num.get(san_move[0], 1)
                encoded_start = self.encode_square(start_square)
                encoded_end = self.encode_square(end_square)
                empty_list.append([piece_num, encoded_start, encoded_end])
        self.encoded_legal_moves = empty_list

    def make_random_move(self):
        legal_moves = list(self.board.legal_moves)
        self.random_move = random.choice(legal_moves)
        return self.random_move

    def get_fen_after_move(self, uci_move):
        move = chess.Move.from_uci(uci_move)
        self.board.push(move)
        fen_after_move = self.board.fen()
        self.board.pop()
        return fen_after_move

    def is_checkmate(self):
        self.checkmate = self.board.is_checkmate()
        if self.checkmate:
            return True, "White" if self.board.turn == chess.WHITE else "Black"
        return False, None

    def step(self, action):
        if isinstance(action, str):
            move = chess.Move.from_uci(action)
        elif isinstance(action, chess.Move):
            move = action
        if move not in self.board.legal_moves:
            raise ValueError("The move is not legal in the current board state.")
        self.board.push(move)
        next_state = chess_board(self.board.fen())
        next_state.checkmate = next_state.board.is_checkmate()
        reward = 1 if next_state.checkmate and next_state.board.turn == chess.BLACK else -1 if next_state.checkmate else 0
        checkmated_player = "White" if next_state.checkmate and next_state.board.turn == chess.BLACK else "Black" if next_state.checkmate else None
        return next_state, reward, next_state.checkmate, checkmated_player

    def stockfish_move(self, elo_rating=10, depth=3):
        if self.stockfish is None:
            raise ValueError("Stockfish not initialized.")
        self.stockfish.set_skill_level(elo_rating)
        self.stockfish.set_depth(depth)
        self.stockfish.set_fen_position(self.FEN)
        try:
            move = self.stockfish.get_best_move()
        except Exception as e:
            print(f"Error while fetching move from engine: {e}")
            move = self.make_random_move()
        return move

    def print_board_image(self, size=400, filename='board_image.png', simulate=False):
        # Generate SVG board image
        svg_board = chess.svg.board(board=self.board, size=size)
        display(SVG(svg_board)) if simulate else None
        return svg_board

    def is_valid_fen(self):
        try:
            chess.Board(self.FEN)
            return True
        except ValueError:
            return False

    def decode_square(self, num):
        file = chr(num % 8 + ord('a'))
        rank = (num // 8) + 1
        return file + str(rank)

    def decode_move(self, tensor):
        flat_tensor = tensor.flatten()
        sorted_values, sorted_indices = torch.sort(flat_tensor, descending=True)
        for i in range(len(sorted_indices)):
            start_square = self.decode_square(sorted_indices[i].item() // 64)
            end_square = self.decode_square(sorted_indices[i].item() % 64)
            if start_square != end_square:
                move = chess.Move.from_uci(start_square + end_square)
                if move in self.board.legal_moves:
                    # print(f"Move found at iteration {i}: {move}")
                    return move