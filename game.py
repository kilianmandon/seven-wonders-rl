from abc import ABC, abstractmethod
import copy
import itertools
import json
import random
from typing import Optional
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

from game_types import ActionSpace, ActionType, GeneralizedColor, PickingAction, PickingActionType, Age, Card, CardColor, CardGroup, CardState, Config, GameState, Player, PlayerScope, PlayerState, ProgressToken, ResourceType, ScienceType, SpecialColors, TerminalState, VictoryType, Wonder, WonderActions




def transition_possible(game_state: GameState, player: PlayerState, card: Card):
    for c1, c2 in game_state.config.transitions:
        if c2 == card and c1 in player.built_cards:
            return True
    return False


def cost_per_resource(player: PlayerState, opponent: PlayerState):
    resource_costs = { resource: 2 for resource in ResourceType }

    for card in opponent.built_cards:
        if card.color in {CardColor.brown, CardColor.gray}:
            for resource in card.resources_provided:
                resource_costs[resource] += 1

    for card in player.built_cards:
        for resource in card.deals_provided:
            resource_costs[resource] = 1

    return resource_costs


def cost_for_card(card: Card | Wonder, player: PlayerState, opponent: PlayerState, free_resources: int=0):
    resource_costs = cost_per_resource(player, opponent)
    resources_to_provide = copy.copy(card.cost_resources)
    resources_to_provide = sorted(resources_to_provide, key=lambda res: resource_costs[res], reverse=True)
    available_resources = []
    
    for built_card in player.built_cards:
        available_resources += built_card.resources_provided
    for wonder in player.built_wonders:
        available_resources += wonder.resources_provided

    for resource in available_resources:
        if isinstance(resource, ResourceType) and resource in resources_to_provide:
            resources_to_provide.remove(resource)

    for resource in available_resources:
        if isinstance(resource, list):
            highest_reduction = 0
            highest_reduction_res = None

            for res in resource:
                cost_reduction = resource_costs[res] if res in resources_to_provide else 0
                if cost_reduction > highest_reduction:
                    highest_reduction = cost_reduction
                    highest_reduction_res = res

            if highest_reduction > 0:
                resources_to_provide.remove(highest_reduction_res)
    
    
    # If n resources are free, remove the first n
    # which are the most expensive ones due to sorting 
    for _ in range(min(free_resources, len(resources_to_provide))):
        resources_to_provide.pop(0)

    cost = sum(resource_costs[res] for res in resources_to_provide) 
    if isinstance(card, Card):
        cost += card.cost_money

    return cost

def is_card_open(game_state: GameState, card_idx: int):
    age_order = game_state.config.age_orders[game_state.age.value]
    return game_state.cards_mask[card_idx] == CardState.revealed and all(game_state.cards_mask[i] == CardState.taken for i in age_order[card_idx])


# For money or victory point computations
def optimal_points_per_building(player: PlayerState, opponent: PlayerState, points_per_building: tuple[int, GeneralizedColor|tuple[GeneralizedColor, GeneralizedColor], PlayerScope]):
    per_building, card_color, scope = points_per_building

    if card_color == SpecialColors.wonder:
        building_count_self = len(player.built_wonders)
        building_count_opponent = len(opponent.built_wonders)
    elif card_color == SpecialColors.three_coins:
        building_count_self = player.money // 3
        building_count_opponent = opponent.money // 3
    else:
        building_count_self = 0
        building_count_opponent = 0
        for card in player.built_cards:
            if card.color == card_color or (isinstance(card_color, list) and card.color in card_color):
                building_count_self += 1

        for card in opponent.built_cards:
            if card.color == card_color or (isinstance(card_color, list) and card.color in card_color):
                building_count_opponent += 1


    if scope == PlayerScope.any_player and building_count_opponent > building_count_opponent:
        return per_building * building_count_opponent
    else:
        return per_building * building_count_self

def construct_card(game_state: GameState, player: PlayerState, opponent: PlayerState, new_card, card_cost):
    if ProgressToken.economy in opponent.progress_tokens:
        # Capped at 0, could be negative if built through transition
        res_card_cost = max(0, card_cost - new_card.cost_money)
        opponent.money += res_card_cost

    if ProgressToken.strategy in player.progress_tokens and new_card.color == CardColor.red:
        player.military_points += 1

    
    player.built_cards.append(new_card)

    player.money += new_card.money_provided - card_cost
    player.victory_points += new_card.victory_points_provided
    player.military_points += new_card.military_points_provided

    if new_card.money_per_building is not None:
        player.money += optimal_points_per_building(player, opponent, new_card.money_per_building)
        

    if new_card.science_provided is not None:
        if new_card.science_provided in player.science and game_state.in_game_progress_tokens:
            game_state.action_space.action_type = ActionType.pick_progress_token
            
        player.science.append(new_card.science_provided)

def pick_card_mask(game_state: GameState, player: PlayerState, opponent: PlayerState):
    mask = np.zeros((20, 14), dtype=bool)
    card_costs = np.zeros((20, 14), dtype=int)
    for i in range(20):
        card = game_state.cards[i]
        if is_card_open(game_state, i):
            mask[i, 0] = True
            card_costs[i, 0] = -2

            free_build_resources = 2 if ProgressToken.masonry in player.progress_tokens and card.color == CardColor.blue else 0
            card_cost = cost_for_card(card, player, opponent, free_build_resources) 
            if card_cost <= player.money or transition_possible(game_state, player, card):
                mask[i, 1] = True
                card_costs[i, 1] = card_cost

            for j, wonder in enumerate(game_state.config.wonders):
                free_wonder_resources = 2 if ProgressToken.architecture in player.progress_tokens else 0
                total_wonder_count = len(player.built_wonders) + len(opponent.built_wonders)
                wonder_cost = cost_for_card(wonder, player, opponent, free_wonder_resources)
                if wonder in player.available_wonders and  wonder_cost <= player.money and total_wonder_count < 7:
                    mask[i, j+2] = True
                    card_costs[i, j+2] = wonder_cost

    mask = mask.flatten()
    card_costs = card_costs.flatten()

    return mask, card_costs


def pick_card(game_state: GameState, player: PlayerState, opponent: PlayerState, card_idx: int, picking_action: PickingAction):
    game_state.second_turn = False

    if not is_card_open(game_state, card_idx):
        raise ValueError('Illegal move: Card is covered')

    action_type = picking_action.picking_action_type
    if action_type == PickingActionType.build_wonder:
        new_wonder = picking_action.wonder
        free_resources = 2 if ProgressToken.architecture in player.progress_tokens else 0
        card_cost = cost_for_card(new_wonder, player, opponent, free_resources)
        total_wonder_count = len(player.built_wonders) + len(opponent.built_wonders)

        if new_wonder not in player.available_wonders:
            raise ValueError('Illegal move: Wonder not available.')
        if total_wonder_count == 7:
            raise ValueError('Illegal move. Already at 7 wonders.')
        if player.money < card_cost:
            raise ValueError(f'Illegal move. Card with cost {card_cost} not affordable, player coins: {player.money}')

        player.available_wonders.remove(new_wonder)
        player.built_wonders.append(new_wonder)
        
        if ProgressToken.economy in opponent.progress_tokens:
            opponent.money += card_cost
            

        player.money += new_wonder.money_provided - card_cost
        opponent.money -= min(opponent.money, new_wonder.money_opponent_lost)
        player.victory_points += new_wonder.victory_points_provided
        player.military_points += new_wonder.military_points_provided

        if new_wonder.second_turn or ProgressToken.theology in player.progress_tokens:
            game_state.second_turn = True

        match new_wonder.action:
            case WonderActions.discard_gray:
                if any(c.color == CardColor.gray for c in opponent.built_cards):
                    game_state.action_space = ActionSpace(action_type=ActionType.discard_opponent_gray)
            case WonderActions.discard_brown:
                if any(c.color == CardColor.brown for c in opponent.built_cards):
                    game_state.action_space = ActionSpace(action_type=ActionType.discard_opponent_brown)
            case WonderActions.random_progress_token:
                game_state.action_space = ActionSpace(action_type=ActionType.pick_discarded_progress_token)
            case WonderActions.choose_discarded:
                if game_state.discarded_cards:
                    game_state.action_space = ActionSpace(action_type=ActionType.pick_discarded_card)


    elif action_type == PickingActionType.build:
        new_card = game_state.cards[card_idx]
        free_build_resources = 2 if ProgressToken.masonry in player.progress_tokens and new_card.color == CardColor.blue else 0
        card_cost = cost_for_card(new_card, player, opponent, free_build_resources)

        if transition_possible(game_state, player, new_card):
            card_cost = 0
            if ProgressToken.urbanism in player.progress_tokens:
                player.money += 4

        if card_cost > player.money:
            raise ValueError(f'Illegal move. Card with cost {card_cost} not affordable, player coins: {player.money}')

        construct_card(game_state, player, opponent, new_card, card_cost)

    else:
        player.money += 2
        game_state.discarded_cards.append(game_state.cards[card_idx])




    game_state.cards_mask[card_idx] = CardState.taken

    update_game_state(game_state)

def pick_progress_token_mask(game_state: GameState):
    mask = [p in game_state.in_game_progress_tokens for p in game_state.config.all_progress_tokens]
    return mask
    

def pick_progress_token(game_state: GameState, player: PlayerState, token_idx: int):
    p = game_state.config.all_progress_tokens[token_idx]
    assert p in game_state.in_game_progress_tokens

    match p:
        case ProgressToken.agriculture:
            player.money += 6
            player.victory_points += 4
        case ProgressToken.law:
            player.science.append(ScienceType.law)
        case ProgressToken.philosophy:
            player.victory_points += 7
        case ProgressToken.urbanism:
            player.money += 6

    player.progress_tokens.append(p)
    game_state.in_game_progress_tokens.remove(p)
    game_state.action_space.action_type = ActionType.pick_card
    update_game_state(game_state)

def discard_opponent_card_mask(game_state: GameState, player: PlayerState, opponent: PlayerState, color: CardColor):
    all_cards: list[Card] = list(itertools.chain(*game_state.config.cards))
    color_cards = [c for c in all_cards if c.color==color]
    mask = [c in opponent.built_cards for c in color_cards]
    return mask
    

def discard_opponent_card(game_state: GameState, player: PlayerState, opponent: PlayerState, color: CardColor, card_idx: int):
    all_cards: list[Card] = list(itertools.chain(*game_state.config.cards))
    color_cards = [c for c in all_cards if c.color==color]
    card = color_cards[card_idx]
    assert card in opponent.built_cards

    opponent.built_cards.remove(card)
    game_state.discarded_cards.append(card)
    game_state.action_space.action_type = ActionType.pick_card
    update_game_state(game_state)


def pick_additional_progress_token_mask(game_state: GameState):
    mask = [p in game_state.additional_progress_tokens for p in game_state.config.all_progress_tokens]
    return mask

def pick_additional_progress_token(game_state: GameState, player: PlayerState, token_idx: int):
    p = game_state.config.all_progress_tokens[token_idx]
    assert p in game_state.additional_progress_tokens
    player.progress_tokens.append(p)
    game_state.action_space.action_type = ActionType.pick_card
    update_game_state(game_state)

def pick_discarded_card_mask(game_state: GameState):
    all_cards: list[Card] = list(itertools.chain(*game_state.config.cards))
    mask = [c in game_state.discarded_cards for c in all_cards]
    return mask

def pick_discarded_card(game_state: GameState, player: PlayerState, opponent: PlayerState, card_idx: int):
    all_cards: list[Card] = list(itertools.chain(*game_state.config.cards))
    card = all_cards[card_idx]
    assert card in game_state.discarded_cards
    game_state.discarded_cards.remove(card)

    game_state.action_space.action_type = ActionType.pick_card
    # Sets game_state to pick_token if necessary
    construct_card(game_state, player, opponent, card, 0)

    player.built_cards.append(card)
    update_game_state(game_state)


def load_cards(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    cards = []
    for entry in data:
        cards.append(Card(**entry))

    return cards

def load_wonders(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    wonders = [Wonder(**entry) for entry in data]
    return wonders

def load_transitions(filename, all_cards):
    with open(filename, 'r') as f:
        data = json.load(f)

    transitions = []
    for (name1, name2) in data:
        card1 = next(card for card in all_cards if card.name==name1)
        card2 = next(card for card in all_cards if card.name==name2)
        transitions.append((card1, card2))
    return transitions

def load_age_order(filename):
    with open(filename, 'r') as f:
        age_orders = json.load(f)
    return age_orders

def load_age_masks(filename):
    with open(filename, 'r') as f:
        age_masks = json.load(f)
    return age_masks

def get_player_and_opponent(game_state: GameState):
    player, opponent = (game_state.player_1_state, game_state.player_2_state) 
    if game_state.active_player == Player.two:
        player, opponent = opponent, player
    return player, opponent
    
    

def load_config():
    age_1_cards = load_cards('card_data/age_1_card_data.json')
    age_2_cards = load_cards('card_data/age_2_card_data.json')
    age_3_cards = load_cards('card_data/age_3_card_data.json')
    guild_cards = load_cards('card_data/guilds_card_data.json')
    wonders = load_wonders('card_data/wonders.json')
    age_orders = load_age_order('card_data/age_orders.json')
    transitions = load_transitions('card_data/transitions.json', age_1_cards+age_2_cards+age_3_cards)
    age_masks = load_age_masks('card_data/age_masks.json')
    all_progress_tokens = list(ProgressToken)

    return Config(age_orders=age_orders, wonders=wonders, cards=[age_1_cards, age_2_cards, age_3_cards, guild_cards], transitions=transitions, age_masks=age_masks, all_progress_tokens=all_progress_tokens)

def load_age(config: Config, age: int, random_initialization=False):
    
    card_pool = config.cards[age]
    if random_initialization:
        k = 20 if age < 2 else 17
        cards = random.sample(card_pool, k)
        if age==2:
            cards += random.sample(config.cards[3], 3)
            random.shuffle(cards)
    else:
        if age == 0:
            card_group = CardGroup.age_one
        elif age == 1:
            card_group = CardGroup.age_two
        else:
            card_group = None
        cards = [
            Card(name='', group=card_group)
         for _ in range(20)]

    card_state = []
    for visible in config.age_masks[age]:
        card_state.append(CardState.revealed if visible else CardState.hidden)

    return cards, card_state


def start_game(random_initialization, seed: Optional[int]=None):
    config = load_config()
    player_1_state = PlayerState()
    player_2_state = PlayerState()
    age = Age.age_one

    if seed is not None:
        random.seed(seed)

    if random_initialization:
        wonders = random.sample(config.wonders, 8)
        player_1_state.available_wonders = wonders[:4]
        player_2_state.available_wonders = wonders[4:]
        progress_tokens = random.sample(config.all_progress_tokens, 5)
        leftover_tokens = [a for a in config.all_progress_tokens if a not in progress_tokens]
        additional_progress_tokens = random.sample(leftover_tokens, 3)

    cards, cards_mask = load_age(config, 0, random_initialization)
    incomplete = False
    for card, mask in zip(cards, cards_mask):
        if card.group is None or (card.name == '' and mask==CardState.revealed):
            incomplete = True


    game_state = GameState(player_1_state=player_1_state, player_2_state=player_2_state, age=age, cards=cards, cards_mask=cards_mask, discarded_cards=[], config=config, random_initialization=random_initialization,
                     active_player=Player.one, incomplete=incomplete, action_space=ActionSpace(action_type=ActionType.pick_card), in_game_progress_tokens=progress_tokens, additional_progress_tokens=additional_progress_tokens)

    update_game_state(game_state, switch_before_picking=False)
    return game_state

def victory_points_evaluation(game_state: GameState, player: PlayerState, opponent: PlayerState):
    victory_points = player.victory_points

    military_difference = player.military_points - opponent.military_points
    if military_difference > 5:
        victory_points += 10
    elif military_difference > 2:
        victory_points += 5
    elif military_difference > 0:
        victory_points += 2

    if ProgressToken.mathematics in player.progress_tokens:
        victory_points += 3 * len(player.progress_tokens)

    for card in player.built_cards:
        if card.victory_points_per_building is not None:
            victory_points += optimal_points_per_building(player, opponent, card.victory_points_per_building)
        
    victory_points += player.money // 3

    blue_points = sum(c.victory_points_provided for c in player.built_cards if c.color==CardColor.blue)

    return victory_points, blue_points


def update_game_state(game_state: GameState, switch_before_picking=True):
    military_diff = game_state.player_1_state.military_points - game_state.player_2_state.military_points

    if military_diff > 8:
        game_state.terminal_state = TerminalState(terminal_state=True, victory_type=VictoryType.military, draw=False, winner=Player.one)
    elif military_diff < -8:
        game_state.terminal_state = TerminalState(terminal_state=True, victory_type=VictoryType.military, draw=False, winner=Player.two)

    if len(set(game_state.player_1_state.science)) >= 6:
        game_state.terminal_state = TerminalState(terminal_state=True, victory_type=VictoryType.science, draw=False, winner=Player.one)
    elif len(set(game_state.player_2_state.science)) >= 6:
        game_state.terminal_state = TerminalState(terminal_state=True, victory_type=VictoryType.science, draw=False, winner=Player.two)
        
    # Important: Only end the game after all cards are taken, if the player would take a card next
    # Actions like discarding cards or picking progress tokens are continued
    if all(m==CardState.taken for m in game_state.cards_mask) and game_state.action_space.action_type==ActionType.pick_card:
        if game_state.age in { Age.age_one, Age.age_two }:
            new_age = Age(game_state.age.value+1)
            cards, card_mask = load_age(game_state.config, new_age.value, game_state.random_initialization)
            game_state.cards = cards
            game_state.cards_mask = card_mask
            game_state.age = new_age
        else:
            p1_points, p1_blue_points = victory_points_evaluation(game_state, game_state.player_1_state, game_state.player_2_state)
            p2_points, p2_blue_points = victory_points_evaluation(game_state, game_state.player_2_state, game_state.player_1_state)

            p1_won = p1_points > p2_points or (p1_points == p2_points and p1_blue_points > p2_blue_points)
            p2_won = p1_points < p2_points or (p1_points == p2_points and p1_blue_points < p2_blue_points)
            if p1_won:
                game_state.terminal_state = TerminalState(is_terminal=True, victory_type=VictoryType.civilian, draw=False, winner=Player.one)
            elif p2_won:
                game_state.terminal_state = TerminalState(is_terminal=True, victory_type=VictoryType.civilian, draw=False, winner=Player.two)
            else:
                game_state.terminal_state = TerminalState(is_terminal=True, victory_type=VictoryType.civilian, draw=True, winner=Player.one)

    if game_state.terminal_state.is_terminal:
        return

    age_order = game_state.config.age_orders[game_state.age.value]
    for i, childs in enumerate(age_order):
        s = game_state.cards_mask[i]
        child_s = [game_state.cards_mask[j] for j in childs]

        if s==CardState.hidden and all(c_s==CardState.taken for c_s in child_s):
            game_state.cards_mask[i] = CardState.revealed

    incomplete_guilds = any(c.group is None for c in game_state.cards)

    incomplete_names = any(m==CardState.revealed and c.name=='' for m,c in zip(game_state.cards_mask, game_state.cards))

    game_state.incomplete = incomplete_guilds or incomplete_names

    if not game_state.incomplete:
        military_diff = game_state.player_1_state.military_points - game_state.player_2_state.military_points

        if military_diff > 2 and not game_state.player_2_state.looted_base:
            game_state.player_2_state.money -= min(2, game_state.player_2_state.money)
            game_state.player_2_state.looted_base = True
        if military_diff > 5 and not game_state.player_2_state.looted_full:
            game_state.player_2_state.money -= min(5, game_state.player_2_state.money)
            game_state.player_2_state.looted_full = True

        if military_diff < -2 and not game_state.player_1_state.looted_base:
            game_state.player_1_state.money -= min(2, game_state.player_1_state.money)
            game_state.player_1_state.looted_base = True
        if military_diff < -5 and not game_state.player_1_state.looted_full:
            game_state.player_1_state.money -= min(5, game_state.player_1_state.money)
            game_state.player_1_state.looted_full = True



        if switch_before_picking and game_state.action_space.action_type ==ActionType.pick_card and not game_state.second_turn:
            if game_state.active_player == Player.one:
                game_state.active_player = Player.two
            else:
                game_state.active_player = Player.one

        player, opponent = get_player_and_opponent(game_state)
        match game_state.action_space.action_type:
            case ActionType.pick_card:
                action_mask, action_costs = pick_card_mask(game_state, player, opponent)
                action_mask = action_mask.tolist()
                action_costs = action_costs.tolist()
                game_state.action_space.action_mask = action_mask
                game_state.action_space.action_cost = action_costs
            case ActionType.pick_progress_token:
                mask = pick_progress_token_mask(game_state)
                game_state.action_space.action_mask = mask
            case ActionType.discard_opponent_gray:
                mask = discard_opponent_card_mask(game_state, player, opponent, CardColor.gray)
                game_state.action_space.action_mask = mask
            case ActionType.discard_opponent_brown:
                mask = discard_opponent_card_mask(game_state, player, opponent, CardColor.brown)
                game_state.action_space.action_mask = mask
            case ActionType.pick_discarded_progress_token:
                mask = pick_additional_progress_token_mask(game_state)
                game_state.action_space.action_mask = mask
            case ActionType.pick_discarded_card:
                mask = pick_discarded_card_mask(game_state)
                game_state.action_space.action_mask = mask

        # game_state_analysis(game_state)


def self_player_state(player: PlayerState, config: Config):
    money_enc = F.one_hot(
        torch.tensor(np.clip(player.money, a_min=0, a_max=20)), num_classes=21)
    science_enc = torch.tensor(
        [1 if p in player.science else 0 for p in ScienceType])
    progress_enc = torch.tensor(
        [1 if p in player.progress_tokens else 0 for p in ProgressToken])
    avail_wonders_enc = torch.tensor(
        [1 if w in player.available_wonders else 0 for w in config.wonders])
    built_wonders = torch.tensor(
        [1 if w in player.built_wonders else 0 for w in config.wonders])

    all_cards = config.all_cards

    built_cards = torch.zeros((len(all_cards,)))
    built_inds = [config.card_name_to_index[c.name] for c in player.built_cards]
    built_cards[built_inds] = 1

    looted_base = torch.tensor([1 if player.looted_base else 0])
    looted_full = torch.tensor([1 if player.looted_full else 0])

    player_enc = torch.cat([
        money_enc, science_enc, progress_enc, avail_wonders_enc, built_wonders, built_cards, looted_base, looted_full
    ])

    return player_enc

def encode_player_state(player: PlayerState, opponent: PlayerState, config: Config):
    military_diff = player.military_points - opponent.military_points
    military_enc = F.one_hot(
        torch.tensor(np.clip(military_diff, a_min=-8, a_max=-8)+8), num_classes=17)

    base_player_state = self_player_state(player, config)
    base_opponent_state = self_player_state(opponent, config)

    full_state = torch.cat([
        base_player_state, base_opponent_state, military_enc
    ])
    return full_state

def encode_game_state(game_state: GameState, player: PlayerState, opponent: PlayerState):
    config = game_state.config
    player_enc = encode_player_state(player, opponent, config)

    all_cards = config.all_cards

    def card_index(i):
        if game_state.cards_mask[i] == CardState.hidden:
            return 0
        elif game_state.cards_mask[i] == CardState.taken:
            return 1
        else:
            return game_state.config.card_name_to_index[game_state.cards[i].name] + 2

    cards_enc = F.one_hot(
        torch.tensor([card_index(i)
                        for i in range(20)]), num_classes=len(all_cards)+2
    )

    card_groups = list(CardGroup)
    card_group_enc = F.one_hot(
        torch.tensor([card_groups.index(c.group)
                        for c in game_state.cards]),
        num_classes=len(card_groups))


    discarded_cards_enc = torch.zeros((len(all_cards,)))
    disc_inds = [game_state.config.card_name_to_index[c.name] for c in game_state.discarded_cards]
    discarded_cards_enc[disc_inds] = 1

    in_game_token_enc = torch.tensor([
        1 if p in game_state.in_game_progress_tokens else 0 for p in ProgressToken
    ])

    additional_progress_tokens_enc = torch.tensor([
        1 if p in game_state.additional_progress_tokens else 0 for p in ProgressToken
    ])

    if game_state.action_space.action_type != ActionType.pick_discarded_progress_token:
        additional_progress_tokens_enc *= 0


    scalar_state = torch.cat([
        player_enc, discarded_cards_enc, in_game_token_enc, 
        additional_progress_tokens_enc
    ])


    board_state = torch.cat([
        cards_enc, card_group_enc
    ], dim=1)

    # Shapes: (386,), (20, 79)
    # return scalar_state, board_state
    return torch.cat([scalar_state, board_state.reshape(-1)])

def create_action_mask(game_state: GameState):
    pc_mask = np.zeros((20*14,))
    ppt_mask = np.zeros((10,))
    dog_mask = np.zeros((4,))
    dob_mask = np.zeros((9,))
    pdp_mask = np.zeros((10,))
    pdc_mask = np.zeros((73,))

    if not game_state.terminal_state.is_terminal:
        match game_state.action_space.action_type:
            case ActionType.pick_card:
                pc_mask = np.array(game_state.action_space.action_mask)
            case ActionType.pick_progress_token:
                ppt_mask = game_state.action_space.action_mask
            case ActionType.discard_opponent_gray:
                dog_mask = game_state.action_space.action_mask
            case ActionType.discard_opponent_brown:
                dob_mask = game_state.action_space.action_mask
            case ActionType.pick_discarded_progress_token:
                pdp_mask = game_state.action_space.action_mask
            case ActionType.pick_discarded_card:
                pdc_mask = game_state.action_space.action_mask

    full_mask = np.concatenate((pc_mask, ppt_mask, dog_mask, dob_mask, pdp_mask, pdc_mask))
    return full_mask

def game_state_analysis(game_state: GameState):
    player, opponent = get_player_and_opponent(game_state)
    s_enc = encode_game_state(game_state, player, opponent)

    mask = torch.tensor(create_action_mask(game_state)).bool()

    n_obs_dim = 1966
    n_actions = 386
    c_h = 256

    policy_net = nn.Sequential(
        nn.Linear(n_obs_dim, c_h),
        nn.ReLU(),
        nn.Linear(c_h, c_h),
        nn.ReLU(),
        nn.Linear(c_h, n_actions),
    )

    policy_net.load_state_dict(torch.load('checkpoints/actor.pt'))
    with torch.no_grad():
        logits = policy_net(s_enc)
        logits[~mask] = -1e6
        probs = F.softmax(logits)

    match game_state.action_space.action_type:
        case ActionType.pick_card:
            action_score = probs[:280]
        case ActionType.pick_progress_token:
            action_score = probs[280:290]
        case ActionType.discard_opponent_gray:
            action_score = probs[290:294]
        case ActionType.discard_opponent_brown:
            action_score = probs[294:303]
        case ActionType.pick_discarded_progress_token:
            action_score = probs[303:313]
        case ActionType.pick_discarded_card:
            action_score = probs[313:386]
    
    game_state.action_space.action_score = action_score.numpy().tolist()
    print(game_state.action_space.action_score)






def main():
    state = start_game(random_initialization=True)
    all_cards = list(itertools.chain(*state.config.cards))
    all_brown = [c for c in all_cards if c.color == CardColor.brown]
    all_gray = [c for c in all_cards if c.color == CardColor.gray]
    print(f'Brown: {len(all_brown)}')
    print(f'Gray: {len(all_gray)}')
    print(f'Total: {len(all_cards)}')
    # TODO: Only three out of five progress tokens are pickable
    # Implement that in state and picking masks

    

if __name__=="__main__":
    main()