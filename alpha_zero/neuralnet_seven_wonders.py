import numpy as np
import torch
from alpha_zero.neuralnet import NeuralNet
from torch import nn
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader
from game import create_action_mask, encode_game_state, get_player_and_opponent
from game_types import GameState, Player


class NeuralNet7Wonders(NeuralNet):
    def __init__(self, game):
        self.n_actions = 386
        self.n_obs = 1966
        c_h = 256
        self.actor = nn.Sequential(
            nn.Linear(self.n_obs, c_h),
            nn.ReLU(),
            nn.Linear(c_h, c_h),
            nn.ReLU(),
            nn.Linear(c_h, self.n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.n_obs, c_h),
            nn.ReLU(),
            nn.Linear(c_h, c_h),
            nn.ReLU(),
            nn.Linear(c_h, 1),
            nn.Tanh(),
        )


    def train(self, examples):
        lr = 1e-3
        epochs = 10
        batch_size = 64

        optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        opposing_player = lambda game_state: Player.one if game_state.active_player == Player.two else Player.two

        states = []
        masks = []
        pis = []
        vs = []

        for (state, mask), pi, v in examples:
            states.append(state)
            masks.append(torch.tensor(mask, dtype=torch.float32))
            pis.append(torch.tensor(pi))
            vs.append(torch.tensor(v, dtype=torch.float32))

        states = torch.stack(states, dim=0)
        masks = torch.stack(masks, dim=0)
        pis = torch.stack(pis, dim=0)
        vs = torch.stack(vs, dim=0).unsqueeze(-1)


        ds = TensorDataset(states, masks, pis, vs)
        dl = DataLoader(ds, batch_size=batch_size)
        pi_loss = nn.CrossEntropyLoss()
        v_loss = nn.MSELoss()

        for batch in dl:
            state, mask, pi, v = batch

            optim_actor.zero_grad()
            optim_critic.zero_grad()

            pi_logits_pred, v_pred = self.actor(state), self.critic(state)
            pi_logits_pred += -1e8 * (1-mask)
            loss = pi_loss(pi_logits_pred, pi) + v_loss(v_pred, v)
            loss.backward()

            optim_actor.step()
            optim_critic.step()




    def predict(self, board: GameState):
        player, opponent = get_player_and_opponent(board)
        s = encode_game_state(board, player, opponent)
        mask = torch.tensor(create_action_mask(board), dtype=torch.float32)
        with torch.no_grad():
            pi_logits = self.actor(s)
            pi_logits += -1e8 * (1-mask)
            pi = F.softmax(pi_logits, dim=-1)
            v = self.critic(s)

        return pi.numpy(), v.numpy()

    def save_checkpoint(self, folder, filename):
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

        torch.save(state_dict, f'{folder}/{filename}')

    def load_checkpoint(self, folder, filename):
        state_dict = torch.load(f'{folder}/{filename}')

        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])


