# game.py

import random
import numpy as np

class Game:
    def __init__(self, seed=None):
        """
        Initialize the game board, score, and game over flag.
        Optionally set a seed for reproducible randomness.
        """
        # Set the seed if provided
        if seed is not None:
            self.random_state = random.Random(seed)
            self.np_random_state = np.random.RandomState(seed)
        else:
            self.random_state = random
            self.np_random_state = np.random

        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.game_over = False

        # Initialize the game with two tiles
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        """
        Add a random tile (2 or 4) to a random empty cell on the board.
        """
        empty_cells = self.get_empty_cells()
        if empty_cells:
            i, j = self.random_state.choice(empty_cells)
            value = self.random_state.choices([2, 4], weights=[0.9, 0.1])[0]
            self.board[i][j] = value

    def get_empty_cells(self):
        """
        Get a list of coordinates for all empty cells on the board.
        """
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
        return empty_cells

    def is_game_over(self):
        """
        Check if the game is over (no moves left).
        """
        if self.get_empty_cells():
            return False
        for i in range(4):
            for j in range(4):
                if (i < 3 and self.board[i][j] == self.board[i + 1][j]) or \
                   (j < 3 and self.board[i][j] == self.board[i][j + 1]):
                    return False
        return True

    def move(self, direction):
        """
        Move the tiles in the specified direction.
        Direction can be 'up', 'down', 'left', or 'right'.
        """
        if direction == 'up':
            self.move_up()
        elif direction == 'down':
            self.move_down()
        elif direction == 'left':
            self.move_left()
        elif direction == 'right':
            self.move_right()
        else:
            print(f"Invalid move direction: {direction}")
            return

        # Add a new random tile after a successful move
        self.add_random_tile()

        # Check if the game is over
        if self.is_game_over():
            self.game_over = True

    def compress(self, board):
        """
        Compress the non-zero tiles in the board to the left.
        """
        new_board = [[0] * 4 for _ in range(4)]
        for i in range(4):
            position = 0
            for j in range(4):
                if board[i][j] != 0:
                    new_board[i][position] = board[i][j]
                    position += 1
        return new_board

    def merge(self, board):
        """
        Merge tiles in the board after compression.
        """
        for i in range(4):
            for j in range(3):
                if board[i][j] != 0 and board[i][j] == board[i][j + 1]:
                    board[i][j] *= 2
                    board[i][j + 1] = 0
                    self.score += board[i][j]
        return board

    def reverse(self, board):
        """
        Reverse the board (used for right and down moves).
        """
        new_board = []
        for i in range(4):
            new_board.append(list(reversed(board[i])))
        return new_board

    def transpose(self, board):
        """
        Transpose the board (used for up and down moves).
        """
        new_board = [[board[j][i] for j in range(4)] for i in range(4)]
        return new_board

    def move_left(self):
        """
        Perform a move to the left.
        """
        new_board = self.compress(self.board)
        new_board = self.merge(new_board)
        self.board = self.compress(new_board)

    def move_right(self):
        """
        Perform a move to the right.
        """
        self.board = self.reverse(self.board)
        self.move_left()
        self.board = self.reverse(self.board)

    def move_up(self):
        """
        Perform a move upwards.
        """
        self.board = self.transpose(self.board)
        self.move_left()
        self.board = self.transpose(self.board)

    def move_down(self):
        """
        Perform a move downwards.
        """
        self.board = self.transpose(self.board)
        self.move_right()
        self.board = self.transpose(self.board)

    def get_available_moves(self):
        """
        Get a list of all possible moves from the current state.
        """
        moves = []
        for direction in ['up', 'down', 'left', 'right']:
            board_copy = [row[:] for row in self.board]
            score_copy = self.score
            self.move(direction)
            if self.board != board_copy:
                moves.append(direction)
            # Restore the original board and score
            self.board = board_copy
            self.score = score_copy
        return moves

    def clone(self):
        """
        Create a deep copy of the game state.
        """
        cloned_game = Game()
        cloned_game.board = [row[:] for row in self.board]
        cloned_game.score = self.score
        cloned_game.game_over = self.game_over
        # Clone random states if necessary
        cloned_game.random_state = self.random_state
        cloned_game.np_random_state = self.np_random_state
        return cloned_game

    def print_board(self):
        """
        Print the current state of the board.
        """
        for row in self.board:
            print('\t'.join(str(num) if num != 0 else '.' for num in row))
        print(f"Score: {self.score}")

