import logging
import numpy as np
from tqdm import tqdm

from gameV2 import Gomaku
from tree_search import TreeSearch

log = logging.getLogger(__name__)

class Arena:
    def __init__(self, player1: TreeSearch, player2: TreeSearch, game: Gomaku, verbose=False):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.verbose = verbose

    def play_game(self):
        # Plays a single game between player1 and player2
        # Returns 1 if player1 won or -1 if player2 won
        players = {
            -1: self.player2,
            1: self.player1
        }
        curr_player = 1
        board = self.game.get_initial_board()
        iterations = 0
        while self.game.game_state(board) is None:  # While game is ongoing
            iterations += 1
            player = players[curr_player]
            player_board = self.game.from_perspective(board, curr_player)

            action = np.argmax(player.get_action_probs(player_board, temp=0))
            valid_moves = self.game.get_valid_moves(player_board)

            if valid_moves[action] == 0:
                raise Exception("Chosen move was invalid")

            board, curr_player = self.game.advance_game(board, curr_player, action)
        return curr_player * self.game.game_state(board)

    def play_round(self, num_games: int):
        # Plays num_games/2 games with player1 starting and num_games/2 games with player2 starting
        num1 = num_games // 2
        num2 = num_games-num1

        player_1_won = 0
        player_2_won = 0
        draws = 0
        for _ in tqdm(range(num1), desc="Arena.playGames (1)"):
            game_result = self.play_game()
            if game_result == 1:
                player_1_won += 1
            elif game_result == -1:
                player_2_won += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num2), desc="Arena.playGames (2)"):
            game_result = self.play_game()
            if game_result == -1:
                player_1_won += 1
            elif game_result == 1:
                player_2_won += 1
            else:
                draws += 1

        return player_1_won, player_2_won, draws