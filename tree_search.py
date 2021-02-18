import logging
import numpy as np

from gameV2 import Gomaku
from agent import Agent

log = logging.getLogger(__name__)

EPS = 1e-8

class TreeSearch:
    """
    Handles monte carlo search tree operations
    """

    def __init__(self, game: Gomaku, agent: Agent, args):
        self.game = game
        self.agent = agent
        self.args = args

        self.q_vals = {}
        self.n_edge_visits = {}
        self.n_state_visits = {}
        self.state_initial_policy = {}

        self.state_game_ended = {}
        self.state_valid_moves = {}

    def get_action_probs(self, board: np.ndarray, temp: int = 1):
        log.debug("Getting action probabilities with board:\n%s", self.game.to_string(board))
        for i in range(self.args["numMCTSSims"]):
            log.debug("Initializing new monte carlo search: %d", i)
            self.search(board)  # Run a bunch of monte carlo simulations

        str_board = self.game.to_string(board)
        # We will use http://ccg.doc.gold.ac.uk/ccg_old/papers/browne_tciaig12_1.pdf page 5 section 3 robust child selection
        counts = np.array([self.n_edge_visits[(str_board, action)] if (str_board, action) in self.n_edge_visits else 0 for action in range(self.game.get_actions_size())])

        if temp == 0:
            # Choose only the best action. This is done at inference time and later in training
            best_action_count = np.max(counts)
            best_actions = np.argwhere(counts == best_action_count).flatten()
            chosen_best_action = np.random.choice(best_actions)
            probs = np.zeros(len(counts))
            probs[chosen_best_action] = 1
            return probs

        # Use a slightly randomized action space
        counts = counts ** (1 / temp)
        probs = counts / np.sum(counts)
        return probs



    def search(self, board: np.ndarray):
        log.debug("Started search")
        str_board = self.game.to_string(board)

        if str_board not in self.state_game_ended:
            # TODO: Make sure that this game over works with the value function
            self.state_game_ended[str_board] = self.game.game_state(board)
        if self.state_game_ended[str_board] is not None:
            return -1*self.state_game_ended[str_board]

        if str_board not in self.state_initial_policy:
            # Then this is a leaf to our tree
            self.state_initial_policy[str_board], state_value = self.agent.predict(board)
            valid_moves = self.game.get_valid_moves(board)
            self.state_initial_policy[str_board] *= valid_moves  # Mask invalid moves
            valid_moves_sum = np.sum(self.state_initial_policy[str_board])
            if valid_moves_sum > 0:
                self.state_initial_policy[str_board] /= valid_moves_sum  # Normalize to a probability distribution
            else:
                # If no move the agent wants to make is valid, make a random valid move.
                log.info("Making random move from: %s", valid_moves)
                self.state_initial_policy[str_board] += valid_moves / np.sum(valid_moves)

            self.state_valid_moves[str_board] = valid_moves
            self.n_state_visits[str_board] = 0  # This is initialized here and updated lower down. We do not count an initial check as a visit
            return -state_value

        valid_moves = self.state_valid_moves[str_board]
        best_q = -float("inf")
        best_action = -1

        for action in np.where(valid_moves == 1)[0]:
            if (str_board, action) in self.q_vals:
                u = self.q_vals[(str_board, action)] + self.args["cpuct"] * self.state_initial_policy[str_board][action] * np.sqrt(self.n_state_visits[str_board]) / (1 + self.n_edge_visits[(str_board, action)])
            else:
                u = 0 + self.args["cpuct"] * self.state_initial_policy[str_board][action] * np.sqrt(self.n_state_visits[str_board] + EPS)

            if u > best_q:
                best_q = u
                best_action = action

        next_board, next_player = self.game.advance_game(board, 1, best_action)
        next_board = self.game.from_perspective(next_board, next_player)

        true_value = self.search(next_board)  # This new board is a flipped old board so the agent learns to play for the white team
        log.debug("Recursive search returned")

        if (str_board, best_action) in self.q_vals:
            # Then we need to update the q value to take the new down-tree value into account
            self.q_vals[(str_board, best_action)] = (self.n_edge_visits[(str_board, best_action)] * self.q_vals[(str_board, best_action)] + true_value) / (self.n_edge_visits[(str_board, best_action)] + 1)
            self.n_edge_visits[(str_board, best_action)] += 1
        else:
            # Then we have no q value to update so we initialize it to the down-tree value
            self.q_vals[(str_board, best_action)] = true_value
            self.n_edge_visits[(str_board, best_action)] = 1

        self.n_state_visits[str_board] += 1
        return -true_value  # The call above this one is for the other player so the value is the negative of this player's value

