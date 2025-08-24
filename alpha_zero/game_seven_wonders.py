import copy
import json
import numpy as np
from alpha_zero.game import Game
from game import create_action_mask, discard_opponent_card, get_player_and_opponent, pick_additional_progress_token, pick_card, pick_discarded_card, pick_progress_token, start_game
from game_types import ActionType, CardColor, CardState, GameState, PickingAction, PickingActionType, Player


class Game7WondersWrapper(Game):
    def __init__(self, store_states=False):
        self.game_state = start_game(random_initialization=True)
        self.store_states = store_states

    def getInitBoard(self):
        self.game_state = start_game(random_initialization=True)
        if self.store_states:
            self.stored_states = [copy.deepcopy(self.game_state)]
        return self.game_state

    def getBoardSize(self):
        return (20, 1)

    def getActionSize(self):
        return 386

    def handle_action(self, game_state, action) -> GameState:
        action_range = np.array([20*14, 10, 4, 9, 10, 73])
        action_cum = np.cumsum(action_range)
        action_area = np.argmax(action < action_cum)
        player, opponent = get_player_and_opponent(game_state)
        if action_area > 0:
            action  -= action_cum[action_area-1]
        if action_area == 0:
            card_idx = action//14
            picking_type = action % 14

            if picking_type==0:
                action_type = PickingAction(picking_action_type = PickingActionType.discard)
            elif picking_type==1:
                action_type = PickingAction(picking_action_type=PickingActionType.build)
            else:
                action_type = PickingAction(picking_action_type=PickingActionType.build_wonder, wonder=game_state.config.wonders[picking_type-2])

            pick_card(game_state, player, opponent, card_idx, action_type)

        elif action_area == 1:
            pick_progress_token(game_state, player, action)

        elif action_area == 2:
            discard_opponent_card(game_state, player, opponent, CardColor.gray, action)

        elif action_area == 3:
            discard_opponent_card(game_state, player, opponent, CardColor.brown, action)

        elif action_area == 4:
            pick_additional_progress_token(game_state, player, action)
        else:
            pick_discarded_card(game_state, player, opponent, action)

        return game_state

    def getNextState(self, board, player, action, save=False):
        lastBoard = copy.deepcopy(board)
        board = self.handle_action(board, action)
        next_player = 1 if board.active_player == Player.one else -1

        if self.store_states and save:
            self.stored_states.append(copy.deepcopy(board))

        return board, next_player

    def store_trajectory(self, i=0):
        if self.store_states:
            states_data = [state.model_dump(mode="json") for state in self.stored_states]
            data = {
                'states': states_data,
                'actions': [None] * len(states_data),
            }
            with open(f'replays/replay_{i}.json', 'w') as f:
                json.dump(data, f)

    def getValidMoves(self, board, player):
        return create_action_mask(board)

    def getGameEnded(self, board: GameState, player):
        if not board.terminal_state.is_terminal:
            return 0

        if board.terminal_state.draw:
            return 1e-5
        
        player_enc = Player.one if player==1 else Player.two

        if board.terminal_state.winner == player_enc:
            return 1
        else:
            return -1


    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board: GameState):
        p1 = self.stringRepresentationPlayer(board.player_1_state)
        p2 = self.stringRepresentationPlayer(board.player_2_state)

        def card_at(i):
            if board.cards_mask[i] == CardState.revealed:
                return board.cards[i].name
            elif board.cards_mask[i] == CardState.hidden:
                return str(board.cards[i].group.value)
            else:
                return 'taken'
        
        card_repr = ';'.join([card_at(i) for i in range(20)])
        disc = ';'.join([c.name for c in sorted(board.discarded_cards, key=lambda c: c.name)])
        prog = ';'.join([c.value for c in board.in_game_progress_tokens])
        if board.action_space.action_type == ActionType.pick_discarded_progress_token:
            prog += '_' + ';'.join([c.value for c in board.additional_progress_tokens])

        rest = f'{board.active_player.value}_{board.age.value}_{board.action_space.action_type.value}'

        full = f'{p1}_{p2}_{card_repr}_{disc}_{prog}_{rest}'

        return full



    def stringRepresentationPlayer(self, player):
        base = f'{str(player.money)}_{str(player.victory_points)}_{str(player.military_points)}'
        science = ';'.join(sorted([s.value for s in player.science]))
        progress = ';'.join(sorted([s.value for s in player.progress_tokens]))
        avail_wonders = ';'.join([s.name for s in sorted(player.available_wonders, key=lambda w: w.name)])
        built_wonders = ';'.join([s.name for s in sorted(player.built_wonders, key=lambda w: w.name)])
        built_cards = ';'.join([s.name for s in sorted(player.built_cards, key=lambda c: c.name)])

        whole = f'{base}_{science}_{progress}_{avail_wonders}_{built_wonders}_{built_cards}_{player.looted_base}_{player.looted_full}'
        return whole