import logging
from torch import nn
from torch.nn import functional as F
import random

import coloredlogs
import numpy as np
import torch

from alpha_zero.arena import Arena
from alpha_zero.coach import Coach
from alpha_zero.mcts import MCTS
from game import create_action_mask, encode_game_state, get_player_and_opponent
from alpha_zero.game_seven_wonders import Game7WondersWrapper

from alpha_zero.neuralnet_seven_wonders import NeuralNet7Wonders
from alpha_zero.utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 800,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 100,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': 'alpha_zero/temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def training():
    log.info('Loading %s...', Game7WondersWrapper.__name__)
    g = Game7WondersWrapper()

    log.info('Loading %s...', NeuralNet7Wonders.__name__)
    nnet = NeuralNet7Wonders(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

def randomMove(game_state):
    action_mask = create_action_mask(game_state)
    options = [i for i, v in enumerate(action_mask) if v]
    return random.choice(options)



def ppo_actor(game_state, policy_net):
    player, opponent = get_player_and_opponent(game_state)
    s_enc = encode_game_state(game_state, player, opponent)

    mask = torch.tensor(create_action_mask(game_state)).bool()

    
    with torch.no_grad():
        logits = policy_net(s_enc)
        logits[~mask] = -1e6
        probs = F.softmax(logits, dim=-1).numpy()

    return np.random.choice(np.arange(probs.size), p=probs)

def eval_against_random():
    g = Game7WondersWrapper(store_states=True)
    pnet = NeuralNet7Wonders(g)
    pnet.load_checkpoint('temp', 'checkpoint_1.pth.tar')
    nnet = NeuralNet7Wonders(g)
    nnet.load_checkpoint('temp', 'checkpoint_15.pth.tar')

    mcts = MCTS(g, nnet, args)
    pmcts = MCTS(g, pnet, args)

    arena = Arena(mcts, ppo_actor, g)
    # p1, p2, draw = arena.playGames(40)
    res = arena.playGame()
    print(res)
    g.store_trajectory(1)
    # print(f'{p1} | {p2} | {draw} (W|L|D)')

    

if __name__ == "__main__":
    training()
