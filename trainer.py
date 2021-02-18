import logging
import numpy as np
from collections import deque
from tqdm import tqdm
from pickle import Pickler, Unpickler
import os

from gameV2 import Gomaku
from agent import Agent
from tree_search import TreeSearch
from arena import Arena

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, game: Gomaku, agent: Agent, args):
        self.game = game
        self.agent = agent
        self.opponent = Agent(self.game)  # Create an agent that we will play against
        self.args = args
        self.tree = TreeSearch(self.game, self.agent, self.args)
        self.past_train_examples = []
        self.skip_first_self_play = False
        self.apply_symmetry = False
        self.curr_player = None

    def play_episode(self):
        log.debug("Starting new episode")
        train_examples = []
        board = self.game.get_initial_board()
        self.curr_player = 1  # While plays first? Doesn't really matter, we can just flip it later.
        episode_step = 0

        game_state = None
        while game_state is None:  # While the game state is ongoing
            episode_step += 1
            # Board is from the perspective of white. Flip it if the opponent is playing
            perspective_board = self.game.from_perspective(board, self.curr_player)
            temp = int(episode_step < self.args["tempThreshold"])

            policy = self.tree.get_action_probs(perspective_board, temp=temp)
            if self.apply_symmetry:
                symmetries = self.game.get_symmetries(perspective_board, policy)
    
                for sym_board, sym_policy in symmetries:
                    # These examples do not yet have a value
                    train_examples.append((sym_board, self.curr_player, sym_policy))
                    # TODO: Check if symmetries are correct
            else:
                train_examples.append((perspective_board, self.curr_player, policy))

            # if game_state is not None:
            #     # We break here so we see the final move
            #     break

            # Choose a random action from the list of possible actions with a probability equal to that actions's prob
            action = np.random.choice(len(policy), p=policy)
            board, self.curr_player = self.game.advance_game(board, self.curr_player, action)

            game_state = self.game.game_state(board)

        final_board = board
        final_policy = [0]*64
        # At this point, the game is over and we know the true value for all actions
        if game_state == 0:
            # Then this game was a draw and all gradients would be 0. No need to train on these examples.
            return []
        training_data = []
        for board, player, policy in train_examples:
            # A 1 game state means the current player won and -1 means current player lost
            value = game_state * (-1 if player == self.curr_player else 1)
            self.log_board(board, policy, value, player)
            training_data.append((board, policy, value))
        self.log_board(final_board, final_policy, game_state*self.curr_player*-1, self.curr_player*-1)
        log.debug("Game ended with %d winning", game_state)
        return training_data

    def log_board(self, board, policy, value, player):
        policy_board = np.reshape(policy, (8, 8))
        board = self.game.to_string(board)
        log.debug(f"\nBoard:\n{board}\n--------\nPolicy:\n{policy_board}\n--------\nValue: {value}, Player: {player}\n        \n")

    def train(self):
        log.info("Starting training")
        for i in range(self.args["numIters"]):
            log.info("Starting iteration: %d", i)
            if not self.skip_first_self_play or i > 0:
                training_data = deque([], maxlen=self.args["maxlenOfQueue"])
                log.info("Starting to play episodes")
                for _ in tqdm(range(self.args["numEps"]), desc="Self Play"):
                    # Recreate the search tree at the current board
                    self.tree = TreeSearch(self.game, self.agent, self.args)
                    training_data += self.play_episode()
                self.past_train_examples.append(training_data)

            log.info("%d training examples available:", len(self.past_train_examples))
            if len(self.past_train_examples) > self.args["numItersForTrainExamplesHistory"]:
                # We have too much data. Pop one
                log.info("Too many past training examples, removing one.")
                self.past_train_examples.pop(0)

            log.info("Finished playing episodes. Saving history")
            self.save_train_history(i)
            train_data = []
            for episode in self.past_train_examples:
                train_data.extend(episode)
            np.random.shuffle(train_data)

            # Load the old network into the opponent for self play test
            self.agent.save_checkpoint(folder=self.args["checkpoint"], filename="temp.pth.tar")
            self.opponent.load_checkpoint(folder=self.args["checkpoint"], filename="temp.pth.tar")
            opponent_tree_search = TreeSearch(self.game, self.opponent, self.args)

            self.agent.train(train_data)
            agent_tree_search = TreeSearch(self.game, self.agent, self.args)

            log.info("Starting self play for evaluation")
            arena = Arena(opponent_tree_search, agent_tree_search, self.game)
            opponent_wins, agent_wins, draws = arena.play_round(self.args["arenaCompare"])
            log.info("Opponent Wins: %d, Agent Wins: %d, Draws: %d", opponent_wins, agent_wins, draws)

            if opponent_wins + agent_wins == 0 or agent_wins / (opponent_wins + agent_wins) < self.args["updateThreshold"]:
                # Then we reject the model as there were all draws or our new model lost too many
                log.info("New model failed to beat old model. Loading old checkpoint.")
                self.agent.load_checkpoint(folder=self.args["checkpoint"], filename="temp.pth.tar")
            else:
                # Then our new agent is better than the last one so we should save it
                # We use i+1 as the index as this is the next model
                log.info("New model beat old model.")
                self.agent.save_checkpoint(folder=self.args["checkpoint"], filename=self.get_checkpoint_filename(i+1))
                self.agent.save_checkpoint(folder=self.args["checkpoint"], filename='best.pth.tar')

    def get_checkpoint_filename(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def save_train_history(self, iteration: int):
        folder = self.args["checkpoint"]
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, self.get_checkpoint_filename(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.past_train_examples)

    def load_train_history(self, iteration: int):
        folder = self.args["checkpoint"]
        filename = os.path.join(folder, self.get_checkpoint_filename(iteration) + ".examples")
        if not os.path.isfile(filename):
            raise Exception("Could not find past examples: " + filename)
        else:
            with open(filename, "rb") as f:
                self.past_train_examples = Unpickler(f).load()
            self.skip_first_self_play = True



