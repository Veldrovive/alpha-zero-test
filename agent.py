import logging
import os
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
from time import time

from agent_net import Network
from gameV2 import Gomaku

log = logging.getLogger(__name__)

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 2
}

class Agent:
    def __init__(self, game: Gomaku):
        self.net = Network(game, args)
        self.board_y, self.board_x = game.get_board_size()
        self.action_size = game.get_actions_size()

        if args["cuda"]:
            self.net.cuda()

    def train(self, examples):
        # Examples is an array of (board, policy, value)
        optimizer = optim.Adam(self.net.parameters())  # TODO: Maybe try using an adabound or something?

        for epoch in range(args["epochs"]):  # TODO: Why is there no randomization here? There's only one shuffle in trainer.py
            log.debug(f"Epoch: {epoch}")
            self.net.train()
            total_policy_loss = 0
            total_value_loss = 0

            batch_count = int(len(examples) / args["batch_size"])

            counter = tqdm(range(batch_count), desc="Training")
            for i in counter:
                sample_ids = np.random.randint(len(examples), size=args["batch_size"]) # Grabs random indeces of examples
                boards, policies, values = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_policies = torch.FloatTensor(np.array(policies))
                target_values = torch.FloatTensor(np.array(values).astype(np.float64))

                if args["cuda"]:
                    boards, target_policies, target_values = boards.contiguous().cuda(), target_policies.contiguous().cuda(), target_values.contiguous().cuda()

                out_policies, out_values = self.net(boards)
                loss_policy = self.policy_loss(target_policies, out_policies)
                loss_value = self.value_loss(target_values, out_values)
                total_loss = loss_policy + loss_value

                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()

                counter.set_postfix(Loss_policy=total_policy_loss/(i+1), Loss_value=total_value_loss/(i+1))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def policy_loss(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def value_loss(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def predict(self, board: np.ndarray):
        start = time()

        board = torch.FloatTensor(board.astype(np.float64))  # TODO: Make sure this doesn't modify game.board in place
        if args["cuda"]:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)  # Reshape? TODO: Figure out why this is 1 deep instead of two splitting up the players like in the paper
        self.net.eval()
        with torch.no_grad():  # No need to compute gradients
            policy, value = self.net(board)

        inference_time = time() - start
        return torch.exp(policy).data.cpu().numpy()[0], value.data.cpu().numpy()[0]

    def save_checkpoint(self, folder="checkpoints", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save({
            'state_dict': self.net.state_dict()
        }, filepath)

    def load_checkpoint(self, folder="checkpoints", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            raise Exception(f"No model found at {filepath}")
        map_location = None if args["cuda"] else 'cpu'  # Move to the correct device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.net.load_state_dict(checkpoint["state_dict"])
