import logging
from typing import List, Tuple, Optional
import numpy as np

log = logging.getLogger(__name__)


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

    def from_string_array(self, board_seed: List[List[str]]):
        # This is the format the game will be delivered in in the competition
        board = np.zeros((self.size, self.size))
        for y in range(self.size):
            for x in range(self.size):
                val = Gomaku.inverse_content[board_seed[y][x]]
                board[y, x] = val
        return board

    def from_string(self, board_seed: str):
        # This is the format that game.to_string returns
        board = np.zeros((self.size, self.size))
        lines = board_seed.split("\n")
        for y in range(len(lines)):
            for x in range(len(lines[0])):
                val = Gomaku.inverse_content[lines[y][x]]
                board[y, x] = val
        return board

    def get_initial_board(self):
        return np.zeros((self.size, self.size))

    def get_symmetries(self, board: np.ndarray, policy: np.ndarray):
        """
        This is a bit confusing. Each board state has 7 other states that represent the exact same value for the
        current player. We can see that any rotation of 90 degrees is a new state that has the exact same value as the
        old one. We can also see if we apply either a vertical or horizontal mirror to this new state we end up with
        another novel state. Combining any more rotations and mirrors just ends up at an already seen state.

        If we neglect these symmetries, we forfeit a simple data augmentation that multiplies our data by 8. So it's
        not really an option to not do this. The numpy is confusing, though.

        board: The (size, size) shape numpy array representing the current board state
        policy: A size*size length numpy array representing the probability of making any given move
        """
        augmented_boards = []
        # We reshape the policy into a board so that when we rotate or reflect it the policy changes correctly
        policy_board = np.reshape(policy, (self.size, self.size))

        for i in [0, 1, 2, 3]:  # For 0, 90, 180, and 270 degrees of rotation
            rot_board = np.rot90(board, i)
            rot_policy_board = np.rot90(policy_board, i)
            flipped_board = np.fliplr(rot_board)
            flipped_policy_board = np.fliplr(rot_policy_board)
            augmented_boards.extend([(rot_board, rot_policy_board.ravel()), (flipped_board, flipped_policy_board.ravel())])
        return augmented_boards

    def __init__(self, size: int):
        self.size = size

    def get_board_size(self) -> Tuple[int, int]:
        return self.size, self.size

    def get_actions_size(self) -> int:
        # The index of action (y, x) is y*self.size + x
        return self.size ** 2

    def get_valid_moves(self, board: np.ndarray) -> np.ndarray:
        # Any location that has a 0 is a valid move
        zeros = np.where(board == 0)
        valid_moves = zeros[0]*self.size + zeros[1]
        all_moves = np.zeros(self.get_actions_size(), dtype=int)
        all_moves[valid_moves] = 1
        return all_moves

    def game_state(self, board: np.ndarray) -> Optional:
        # Returns -1 is black won and 1 if white won. 0 if there was a draw. None if the game is ongoing.
        # TODO: Check if there is a valid 5 in a row
        def detect_five(y_start, x_start, d_y, d_x):
            seq_length = None
            seq_player = None
            y_max = self.size - 1
            x_max = self.size - 1
            while 0 <= y_start <= y_max and 0 <= x_start <= x_max:
                cur_player = board[y_start, x_start]
                if cur_player == seq_player:
                    seq_length += 1
                else:
                    if seq_length == 5:
                        return seq_player
                    seq_length = None
                    seq_player = None
                    if cur_player != 0:
                        seq_length = 1
                        seq_player = cur_player
                y_start += d_y
                x_start += d_x
            return None

        # Check for a winner
        winner = None
        x_start = 0
        for y_start in range(len(board)):
            # Check directions (0, 1) and (1, 1)
            winner = winner or detect_five(y_start, x_start, 0, 1)
            winner = winner or detect_five(y_start, x_start, 1, 1)
        x_start = len(board[0]) - 1
        for y_start in range(len(board)):
            # Check direction (1, -1)
            winner = winner or detect_five(y_start, x_start, 1, -1)
        y_start = 0
        for x_start in range(len(board[0])):
            # Check direction (1, 0)
            winner = winner or detect_five(y_start, x_start, 1, 0)
            if x_start > 1:
                # Check the rows that were not on the y pass
                winner = winner or detect_five(y_start, x_start, 1, 1)
            if x_start < len(board[0]) - 1:
                # Chck the rows that were not on the second y pass
                winner = winner or detect_five(y_start, x_start, 1, -1)
        if winner is not None:
            return winner

        if np.max(self.get_valid_moves(board)) < 1:
            return 0
        return None

    def advance_game(self, board: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, int]:
        # Plays a piece onto the board
        action_y, action_x = action // self.size, action % self.size
        new_board = board.copy()
        new_board[action_y, action_x] = player
        return new_board, -player

    def from_perspective(self, board: np.ndarray, player: int) -> np.ndarray:
        # If player is -1 then the board will be flipped such that the player -1 has the white pieces
        return player*board

    def to_string(self, board: np.ndarray) -> str:
        # Converts the game board to a string
        return '\n'.join([''.join([self.content[x] for x in line]) for line in board])