from typing import List
import numpy as np


class Gomaku:
    content = {
        -1: "b",
        0: " ",
        1: "w"
    }

    inverse_content = {
        "b": -1,
        " ": 0,
        "1": 1
    }

    @staticmethod
    def from_string_array(board: List[List[str]]):
        size = len(board)
        game = Gomaku(size)
        for y in range(size):
            for x in range(size):
                val = Gomaku.inverse_content[board[y][x]]
                game.board[y, x] = val
        return game

    @staticmethod
    def from_string(board: str):
        lines = board.split("\n")
        game = Gomaku(len(lines))
        for y in range(len(lines)):
            for x in range(len(lines[0])):
                val = Gomaku.inverse_content[lines[y][x]]
                game.board[y, x] = val
        return game

    def __init__(self, size: int):
        self.size = size
        self.board = np.zeros((size, size), int)

    def get_board_size(self):
        return self.size, self.size

    def get_actions_size(self):
        # The index of action (y, x) is y*self.size + x
        return self.size ** 2

    def get_valid_moves(self):
        # Any location that has a 0 is a valid move
        zeros = np.where(self.board == 0)
        valid_moves = zeros[0]*self.size + zeros[1]
        all_moves = np.zeros(self.get_actions_size(), dtype=int)
        all_moves[valid_moves] = 1
        return all_moves

    def game_over(self):
        # Returns -1 is black won and 1 if white won. 0 if there was a draw. None if the game is ongoing.
        # TODO: Check if there is a valid 5 in a row
        if len(self.get_valid_moves()) == 0:
            return 0
        return None

    def advance_game(self, player: int, action: int):
        # Plays a piece onto the board
        action_y, action_x = action // self.size, action % self.size
        self.board[action_y, action_x] = player
        return self.board, -player

    def __str__(self):
        # Converts the game board to a string
        return '\n'.join([''.join([self.content[x] for x in line]) for line in self.board])



if __name__ == "__main__":
    print("Running game unit tests")
    board_size = 8
    g1 = Gomaku(board_size)
    size = g1.get_board_size()
    actions = g1.get_actions_size()
    valid_moves = g1.get_valid_moves()
    print(f"Size: {size} - Number of actions: {actions}")
    print(f"Valid actions: {valid_moves}")
    print(f"Game board:\n{g1}")
    assert size == (board_size, board_size)
    assert actions == board_size*board_size
    assert len(valid_moves) == actions

    g1.advance_game(-1, 1*board_size+4)
    print(f"Game board:\n{g1}")
    valid_moves = g1.get_valid_moves()
    print(f"Valid actions: {valid_moves}")
    assert 1*board_size+4 not in valid_moves

    print(f"Valid Actions: {np.where(valid_moves == 1)[0]}")
