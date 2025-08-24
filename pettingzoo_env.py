import copy
import datetime
import functools
import itertools
import json
import random
import numpy as np
from pettingzoo import AECEnv
from gymnasium.spaces import Discrete, Box
from gymnasium import spaces
import torch
from torch.nn import functional as F
import tqdm

from game import create_action_mask, discard_opponent_card, encode_game_state, get_player_and_opponent, pick_additional_progress_token, pick_card, pick_discarded_card, pick_progress_token, start_game
from game_types import ActionType, Card, CardColor, CardGroup, CardState, Config, GameState, PickingAction, PickingActionType, Player, PlayerState, ProgressToken, ScienceType


class GameEnv(AECEnv):
    metadata = {
        "name": "7wonders",
    }
    def __init__(self, store_states=False):
        # gym.spaces.Dict({
        #     'pick_card': gym.spaces.Discrete(20*14),
        #     'pick_progress_token': gym.spaces.Discrete(10),
        #     'discard_opponent_gray': gym.spaces.Discrete(4),
        #     'discard_opponent_brown': gym.spaces.Discrete(9),
        #     'pick_discarded_progress_token': gym.spaces.Discrete(10),
        #     'pick_discarded_card': gym.spaces.Discrete(73),
        # })
        # pick_card + pick_progress_token + discard_opponent_gray +
        #    discard_opponent_brown + pick_discarded_progress_token + pick_discarded_card
        self.n_actions = 20*14 + 10 + 4 + 9 + 10 + 73
        self.possible_agents = [f"player_{i}" for i in range(2)]
        self._action_spaces = {agent: Discrete(
            self.n_actions) for agent in self.possible_agents}

        self.store_states = store_states
        if self.store_states:
            self.stored_states = []
            self.stored_actions = []

    def _get_obs(self):
        ...



    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({"observation": Box(0, 1, (386+20*79,)), "action_mask": Box(0, 1, (self.n_actions,))})

        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.n_actions)

    def observe(self, agent):
        return self.observations[agent]

    def close(self):
        if self.store_states:
            states_data = [state.model_dump(mode="json") for state in self.stored_states]
            data = {
                'states': states_data,
                'actions': self.stored_actions,
            }
            date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            with open(f'replays/replay_{date_time_str}.json', 'w') as f:
                json.dump(data, f)

    def reset(self, seed=None, options=None):
        self.game_state = start_game(True, seed)
        self.agents = self.possible_agents
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.agent_selection = self.possible_agents[0]
        self.num_moves = 0

        self.set_observations(self.game_state)

        if self.store_states:
            self.stored_states = [self.game_state]

    def handle_action(self, game_state, action):
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

    def set_observations(self, game_state):
        full_mask = create_action_mask(game_state)

        if game_state.active_player == Player.one:
            p1_mask = full_mask
            p2_mask = np.zeros_like(full_mask)
        else:
            p1_mask = np.zeros_like(full_mask)
            p2_mask = full_mask

        self.observations["player_0"] = {
            "observation": encode_game_state(game_state, game_state.player_1_state, game_state.player_2_state),
            "action_mask": p1_mask
        }

        self.observations["player_1"] = {
            "observation": encode_game_state(game_state, game_state.player_2_state, game_state.player_1_state),
            "action_mask": p2_mask
        }


    def step(self, action):
        action = action.item()
        agent = self.agent_selection
        if self.store_states:
            new_game_state = copy.deepcopy(self.game_state)
        else:
            new_game_state = self.game_state
        new_game_state = self.handle_action(new_game_state, action)
        self.set_observations(new_game_state)



        if new_game_state.terminal_state.is_terminal:
            if not new_game_state.terminal_state.draw:
                if new_game_state.terminal_state.winner == Player.one:
                    self.rewards["player_0"] = 1
                    self.rewards["player_1"] = -1
                else:
                    self.rewards["player_1"] = 1
                    self.rewards["player_0"] = -1

            self.terminations = {
                f"player_{i}": True for i in range(2)
            }

        self.agent_selection = "player_0" if new_game_state.active_player==Player.one else "player_1"

        self._accumulate_rewards()

        ## Important!
        self.game_state = new_game_state
        if self.store_states:
            self.stored_actions.append(action)
            self.stored_states.append(new_game_state)


    



if __name__=='__main__':
    env = GameEnv(store_states=False)

    env.reset()

    for i in tqdm.tqdm(range(10000)):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            if termination or truncation:
                action = None
                break
            else:
                action_mask = observation["action_mask"]
                action = random.choice([i for i, m in enumerate(action_mask) if m])

            env.step(action)

    env.close()


        

        
