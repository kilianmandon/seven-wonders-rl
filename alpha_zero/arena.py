import logging
import numpy as np
from torch.multiprocessing import Pool

from tqdm import tqdm

from alpha_zero.mcts import MCTS

log = logging.getLogger(__name__)

def playGame(args):
    """
    Executes one episode of a game.

    Returns:
        either
            winner: player who won the game (1 if player1, -1 if player2)
        or
            draw result returned from the game that is neither 1, -1, nor 0.
    """
    player1, player2, game = args
    players = [player2, None, player1]
    curPlayer = 1
    board = game.getInitBoard()
    it = 0

    for player in players[0], players[2]:
        if hasattr(player, "startGame"):
            player.startGame()

    while game.getGameEnded(board, curPlayer) == 0:
        it += 1
        # action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer))
        
        if isinstance(players[curPlayer + 1], MCTS):
            action = np.argmax(players[curPlayer+1].getActionProb(game.getCanonicalForm(board, curPlayer), temp=0))
        else:
            action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer))

        valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1)

        if valids[action] == 0:
            log.error(f'Action {action} is not valid!')
            log.debug(f'valids = {valids}')
            assert valids[action] > 0

        # Notifying the opponent for the move
        opponent = players[-curPlayer + 1]
        if hasattr(opponent, "notify"):
            opponent.notify(board, action)

        board, curPlayer = game.getNextState(board, curPlayer, action, save=True)

    for player in players[0], players[2]:
        if hasattr(player, "endGame"):
            player.endGame()

    return curPlayer * game.getGameEnded(board, curPlayer)

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)

            if isinstance(players[curPlayer + 1], MCTS):
                action = np.argmax(players[curPlayer+1].getActionProb(board, temp=0))
            else:
                action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))



            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, curPlayer = self.game.getNextState(board, curPlayer, action, save=True)

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        with Pool() as executor:
            results = list(tqdm(
                executor.imap_unordered(playGame, [(self.player1, self.player2, self.game)]*num),
                total=num,
                desc="Arena.playGames (1)"
            ))

        for gameResult in results:
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        with Pool() as executor:
            results = list(tqdm(
                executor.imap_unordered(playGame, [(self.player1, self.player2, self.game)]*num),
                total=num,
                desc="Arena.playGames (2)"
            ))

        for gameResult in results:
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
