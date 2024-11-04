# main.py

import tkinter as tk
import time
import argparse
import random
import numpy as np
from game import Game
from ai import AI
from visualize import Visualizer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the 2048 AI with an optional seed.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random number generators.')
    args = parser.parse_args()

    # Set the seed if provided
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
    else:
        seed = None  # No seed provided; randomness will be unpredictable

    # Load the best weights
    best_weights = [0.88492502, 0.46450413, 6.07544852, 1.82431396]  # Replace with your actual best weights

    # Initialize the game, AI, and visualization
    game = Game(seed=seed)
    ai = AI(weights=best_weights, search_depth=3, seed=seed)  # Adjust search_depth if desired
    root = tk.Tk()
    visualizer = Visualizer()
    visualizer.master = root
    visualizer.pack()

    # Display the initial game state
    visualizer.display(game)

    # Set up the game loop
    def game_loop():
        if not game.is_game_over():
            # AI decides the move
            move = ai.decide_move(game)
            if move:
                game.move(move)
                visualizer.display(game)
            else:
                print("No valid moves available.")
            # Schedule the next move
            visualizer.after(100, game_loop)  # Adjust delay as needed (in milliseconds)
        else:
            # Game over
            visualizer.display_game_over(game)
            print("Game Over!")
            print(f"Final Score: {game.score}")
            # Optionally, print the highest tile achieved
            max_tile = max(max(row) for row in game.board)
            print(f"Highest Tile: {max_tile}")

    # Start the game loop
    visualizer.after(100, game_loop)
    root.mainloop()

if __name__ == "__main__":
    main()
