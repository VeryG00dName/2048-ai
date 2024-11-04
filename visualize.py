# visualize.py

import tkinter as tk
from tkinter import Frame, Label, CENTER

# Define colors for the tiles
BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"

# Dictionary of colors for different tile values
BACKGROUND_COLOR_DICT = {
    0:      "#9e948a",
    2:      "#eee4da",
    4:      "#ede0c8",
    8:      "#f2b179",
    16:     "#f59563",
    32:     "#f67c5f",
    64:     "#f65e3b",
    128:    "#edcf72",
    256:    "#edcc61",
    512:    "#edc850",
    1024:   "#edc53f",
    2048:   "#edc22e",
    4096:   "#3c3a32",
    8192:   "#3c3a32",
    16384:  "#3c3a32",
    32768:  "#3c3a32",
    65536:  "#3c3a32",
}

CELL_COLOR_DICT = {
    0:      "#9e948a",
    2:      "#776e65",
    4:      "#776e65",
    8:      "#f9f6f2",
    16:     "#f9f6f2",
    32:     "#f9f6f2",
    64:     "#f9f6f2",
    128:    "#f9f6f2",
    256:    "#f9f6f2",
    512:    "#f9f6f2",
    1024:   "#f9f6f2",
    2048:   "#f9f6f2",
    4096:   "#f9f6f2",
    8192:   "#f9f6f2",
    16384:  "#f9f6f2",
    32768:  "#f9f6f2",
    65536:  "#f9f6f2",
}

FONT = ("Verdana", 40, "bold")
SCORE_FONT = ("Verdana", 24)

class Visualizer(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.grid()
        self.master.title('2048')
        self.master.resizable(False, False)
        self.grid_cells = []
        self.init_grid()
        self.update_idletasks()

    def init_grid(self):
        """Initialize the GUI grid of cells."""
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=500, height=500)
        background.grid(pady=(100, 0))
        for i in range(4):
            grid_row = []
            for j in range(4):
                cell = Frame(
                    background,
                    bg=BACKGROUND_COLOR_CELL_EMPTY,
                    width=125,
                    height=125
                )
                cell.grid(row=i, column=j, padx=5, pady=5)
                label = Label(
                    master=cell,
                    text="",
                    bg=BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=FONT,
                    width=4,
                    height=2
                )
                label.grid()
                grid_row.append(label)
            self.grid_cells.append(grid_row)

        # Score Display
        self.score_label = Label(
            self,
            text="Score: 0",
            font=SCORE_FONT,
            fg="#ffffff",
            bg=BACKGROUND_COLOR_GAME
        )
        self.score_label.grid(sticky="n")

    def display(self, game):
        """Update the GUI to reflect the current game state."""
        for i in range(4):
            for j in range(4):
                tile_value = game.board[i][j]
                if tile_value == 0:
                    self.grid_cells[i][j].configure(
                        text="",
                        bg=BACKGROUND_COLOR_CELL_EMPTY
                    )
                else:
                    self.grid_cells[i][j].configure(
                        text=str(tile_value),
                        bg=BACKGROUND_COLOR_DICT.get(tile_value, BACKGROUND_COLOR_DICT[65536]),
                        fg=CELL_COLOR_DICT.get(tile_value, CELL_COLOR_DICT[65536])
                    )
        self.score_label.configure(text=f"Score: {game.score}")
        self.update_idletasks()

    def display_game_over(self, game):
        """Display the game over screen."""
        self.display(game)
        game_over_frame = Frame(self, borderwidth=0)
        game_over_frame.place(relx=0.5, rely=0.5, anchor=CENTER)
        Label(
            game_over_frame,
            text="Game Over!",
            font=("Verdana", 48, "bold"),
            fg="#ffffff",
            bg="#000000",
            justify=CENTER
        ).pack()
        Label(
            game_over_frame,
            text=f"Final Score: {game.score}",
            font=SCORE_FONT,
            fg="#ffffff",
            bg="#000000",
            justify=CENTER
        ).pack()
        self.update_idletasks()
