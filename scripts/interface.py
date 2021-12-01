#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
from tkinter import *

# PROJECT MODULES


# GLOBAL CONSTANTS
WIDTH = 900
HEIGHT = 500
WINDOW_SIZE = f"{WIDTH}x{HEIGHT}"
BACKGROUND_COLOR = '#3b414a'
TEXT_COLOR = '#121212'

M = 5
N = 20


# FUNCTIONS
def window_config(window, title):
    x = int((SCREEN_WIDTH - WIDTH) / 2)
    y = int((SCREEN_HEIGHT - HEIGHT) / 2)
    window.geometry(f'{WIDTH}x{HEIGHT}+{x}+{y}')
    window.title(title)
    window.config(background=BACKGROUND_COLOR)

    return window



class Cells:
    def __init__(self, frame, width):
        self.cells = []

        for i in range(M):
            cells = []
            for j in range(N):
                textarea = Entry(frame,
                                 fg=TEXT_COLOR,
                                 bg="white",
                                 width=width // (N * 2),
                                 state="normal",
                                 disabledbackground='#000000')
                textarea.grid(row=i, column=j)
                textarea.bind("<Up>", lambda event: self.move_cursor_up(self.cells[min(i - 1, 0)][j]))
                textarea.bind("<Down>", lambda event: self.move_cursor_down(max(i + 1, M - 1), j))
                textarea.bind("<Right>", lambda event: self.move_cursor_right(i, min(j + 1, N - 1)))
                textarea.bind("<Left>", lambda event: self.move_cursor_left(i, max(j - 1, 0)))
                cells.append(textarea)

            self.cells.append(cells)

    def move_cursor_up(self, cell):
        print("Up")
        cell.icursor("end")

    def move_cursor_down(self, i, j):
        pass

    def move_cursor_right(self, i, j):
        pass

    def move_cursor_left(self, i, j):
        pass

if __name__ == "__main__":
    root = Tk()
    SCREEN_WIDTH = root.winfo_screenwidth()
    SCREEN_HEIGHT = root.winfo_screenheight()

    root = window_config(root, 'CostShare')
    frame = Frame(root)
    frame.pack()
    c = Cells(frame, SCREEN_WIDTH)
    root.mainloop()
