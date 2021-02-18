import logging

import coloredlogs

from gameV2 import Gomaku
from agent import Agent
from trainer import Trainer

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info or INFO to see less.

args = {
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_file': 'best.pth.tar',
    'numItersForTrainExamplesHistory': 20,

}

def main():
    game = Gomaku(8)
    agent = Agent(game)

    if args["load_model"]:
        log.info("Loading old model: %s/%s", args["checkpoint"], args["load_file"])
        agent.load_checkpoint(folder=args["checkpoint"], filename=args["load_file"])
    else:
        log.info("Creating new model")

    trainer = Trainer(game, agent, args)

    log.info("Starting training")
    trainer.train()


if __name__ == "__main__":
    main()

