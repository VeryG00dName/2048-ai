# ai.py

import time
import random
import numpy as np

class AI:
    def __init__(self, weights, search_depth=5, time_limit=5.0, seed=None):
        """
        Initialize the AI with given weights and search parameters.

        :param weights: A list or array of weights for the evaluation function.
        :param search_depth: The maximum depth to search in the expectimax algorithm.
        :param time_limit: The maximum time allowed for making a move (in seconds).
        """
        self.weights = weights  # Weights for the evaluation function
        self.search_depth = search_depth
        self.time_limit = time_limit
        if seed is not None:
            self.random_state = random.Random(seed)
            self.np_random_state = np.random.RandomState(seed)
        else:
            self.random_state = random
            self.np_random_state = np.random

    def decide_move(self, game):
        """
        Decide the best move by using the expectimax algorithm.

        :param game: The current game state.
        :return: The best move as a string ('up', 'down', 'left', 'right').
        """
        start_time = time.time()
        _, best_move = self.expectimax(game, depth=self.search_depth, is_player_turn=True, start_time=start_time)
        return best_move

    def expectimax(self, game, depth, is_player_turn, start_time):
        """
        The expectimax search algorithm.

        :param game: The current game state.
        :param depth: The current depth in the search tree.
        :param is_player_turn: True if it's the player's turn, False if it's the computer's turn.
        :param start_time: The time when the search started.
        :return: A tuple of (score, move), where move is only relevant for the root call.
        """
        # Time limit check
        if time.time() - start_time >= self.time_limit:
            return self.evaluate(game), None

        # Base case: depth limit reached or game over
        if depth == 0 or game.is_game_over():
            return self.evaluate(game), None

        if is_player_turn:
            # Player's turn: Maximize the score
            max_score = float('-inf')
            best_move = None
            for move in game.get_available_moves():
                new_game = game.clone()
                new_game.move(move)
                score, _ = self.expectimax(new_game, depth - 1, False, start_time)
                if score > max_score:
                    max_score = score
                    best_move = move
            return max_score, best_move
        else:
            # Computer's turn: Expectation over possible tile spawns
            scores = []
            empty_cells = game.get_empty_cells()
            if not empty_cells:
                return self.evaluate(game), None
            for cell in empty_cells:
                for value, probability in [(2, 0.9), (4, 0.1)]:
                    new_game = game.clone()
                    i, j = cell
                    new_game.board[i][j] = value
                    score, _ = self.expectimax(new_game, depth - 1, True, start_time)
                    scores.append(score * probability)
            expected_score = sum(scores) / len(scores)
            return expected_score, None

    def evaluate(self, game):
        """
        Evaluate the desirability of the game state using the weighted evaluation function.

        :param game: The current game state.
        :return: The evaluated score.
        """
        board = game.board
        score = 0

        # Extract weights
        # Adjust the number of weights and their usage based on your evaluation function
        corner_weight, adjacency_weight, empty_cells_weight, monotonicity_weight = self.weights

        # 1. Highest Tile in the Corner (Positive heuristic)
        max_tile = max(max(row) for row in board)
        corner_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
        in_corner = any(board[i][j] == max_tile for i, j in corner_positions)
        corner_score = max_tile if in_corner else 0
        score += corner_weight * corner_score

        # 2. Adjacent tiles with the same value
        adjacent_score = self.calculate_adjacent_tiles(board)
        score += adjacency_weight * adjacent_score

        # 3. Number of empty cells
        empty_cells = len(game.get_empty_cells())
        score += empty_cells_weight * empty_cells

        # 4. Monotonicity
        monotonicity = self.calculate_monotonicity(board)
        score += monotonicity_weight * monotonicity

        return score

    def calculate_adjacent_tiles(self, board):
        """
        Calculate a score based on adjacent tiles with the same value.

        :param board: The game board.
        :return: The adjacent tiles score.
        """
        score = 0
        for i in range(4):
            for j in range(4):
                current_value = board[i][j]
                if current_value == 0:
                    continue
                # Check right and down neighbors
                for dx, dy in [(0, 1), (1, 0)]:
                    x, y = i + dx, j + dy
                    if 0 <= x < 4 and 0 <= y < 4:
                        neighbor_value = board[x][y]
                        if neighbor_value == current_value:
                            score += current_value
        return score

    def calculate_monotonicity(self, board):
        """
        Calculate a score based on the monotonicity of the board.

        :param board: The game board.
        :return: The monotonicity score.
        """
        totals = [0, 0, 0, 0]  # Up, Down, Left, Right

        # Rows
        for i in range(4):
            current_row = board[i]
            for j in range(3):
                if current_row[j] > current_row[j + 1]:
                    totals[0] += current_row[j + 1] - current_row[j]
                elif current_row[j] < current_row[j + 1]:
                    totals[1] += current_row[j] - current_row[j + 1]

        # Columns
        for j in range(4):
            current_col = [board[i][j] for i in range(4)]
            for i in range(3):
                if current_col[i] > current_col[i + 1]:
                    totals[2] += current_col[i + 1] - current_col[i]
                elif current_col[i] < current_col[i + 1]:
                    totals[3] += current_col[i] - current_col[i + 1]

        # Return the maximum monotonicity
        return max(totals)

    # Additional methods can be added here if needed
