from enum import Enum

from pydantic import BaseModel


class PlayerScope(Enum):
    self_player = "self"
    any_player = "any"

class CardColor(Enum):
    brown = "brown"
    gray = "gray"
    red = "red"
    green = "green"
    blue = "blue"
    yellow = "yellow"
    purple = "purple"

class CardGroup(Enum):
    age_one = 0
    age_two = 1
    age_three = 2
    guild = 3



class SpecialColors(Enum):
    wonder = "wonder"
    three_coins = "three_coins"

GeneralizedColor = CardColor | SpecialColors

class ResourceType(Enum):
    wood = "wood"
    clay = "clay"
    stone = "stone"
    glass = "glass"
    papyrus = "papyrus"

class ScienceType(Enum):
    astrolabe = "astrolabe"
    scales = "scales"
    solar_clock = "solar_clock"
    mortar = "mortar"
    plumb_triangle = "plumb_triangle"
    quill = "quill"
    wheel = "wheel"
    law = "law"

class ProgressToken(Enum):
    agriculture = "agriculture"
    architecture = "architecture"
    economy = "economy"
    law = "law"
    masonry = "masonry"
    mathematics = "mathematics"
    philosophy = "philosophy"
    strategy = "strategy"
    theology = "theology"
    urbanism = "urbanism"

class WonderActions(Enum):
    discard_gray = "discard_gray"
    random_progress_token = "random_progress_token"
    choose_discarded = "choose_discarded"
    discard_brown = "discard_brown"

class Wonder(BaseModel):
    name: str
    cost_resources: list[ResourceType]
    resources_provided: list[ResourceType | list[ResourceType]]
    military_points_provided: int
    victory_points_provided: int
    money_provided: int
    money_opponent_lost: int
    second_turn: bool
    action: WonderActions | None

class Card(BaseModel):
    name: str = ''
    group: CardGroup = None
    color: CardColor = None
    cost_resources: list[ResourceType] = []
    cost_money: int = 0
    resources_provided: list[ResourceType | list[ResourceType]] = []
    military_points_provided: int = 0
    victory_points_provided: int = 0
    science_provided: ScienceType | None = None
    deals_provided: list[ResourceType] = []
    money_provided: int = 0
    money_per_building: tuple[int, GeneralizedColor|tuple[GeneralizedColor, GeneralizedColor], PlayerScope] | None = None
    victory_points_per_building: tuple[int, GeneralizedColor|tuple[GeneralizedColor, GeneralizedColor], PlayerScope] | None = None

class Config(BaseModel):
    age_orders: list[list[list[int]]]
    wonders: list[Wonder]
    cards: list[list[Card]]
    transitions: list[tuple[Card, Card]]
    age_masks: list[list[bool]]
    all_progress_tokens: list[ProgressToken]

    @property
    def all_cards(self) -> list[Card]:
        return [c for row in self.cards for c in row]

    @property
    def card_name_to_index(self) -> dict[str, int]:
        return {card.name: i for i, card in enumerate(self.all_cards)}

    @property
    def index_to_card_name(self) -> dict[int, str]:
        return {i: card.name for i, card in enumerate(self.all_cards)}

class PlayerState(BaseModel):
    money: int = 0
    victory_points: int = 0
    military_points: int = 0
    science: list[ScienceType] = []
    progress_tokens: list[ProgressToken] = []

    available_wonders: list[Wonder] = []
    built_wonders: list[Wonder] = []
    built_cards: list[Card] = []
    looted_base: bool = False
    looted_full: bool = False

class Age(Enum):
    age_one = 0
    age_two = 1
    age_three = 2

class CardState(Enum):
    taken = "taken"
    revealed = "revealed"
    hidden = "hidden"

class Player(Enum):
    one = 0
    two = 1

class ActionType(Enum):
    no_action_required = 0
    pick_card = 1
    pick_progress_token = 2
    discard_opponent_gray = 3
    discard_opponent_brown = 4
    pick_discarded_progress_token = 5
    pick_discarded_card = 6

class ActionSpace(BaseModel):
    action_type: ActionType = ActionType.no_action_required
    action_mask: list[bool] = []
    action_cost: list[int] = []
    action_score: list[float] = []

class VictoryType(Enum):
    military = "military"
    science = "science"
    civilian = "civilian"



class TerminalState(BaseModel):
    is_terminal: bool = False
    victory_type: VictoryType = VictoryType.civilian
    draw: bool = False
    winner: Player = Player.one


class GameState(BaseModel):
    player_1_state: PlayerState
    player_2_state: PlayerState
    age: Age
    cards: list[Card]
    cards_mask: list[CardState]
    discarded_cards: list[Card]
    in_game_progress_tokens: list[ProgressToken]
    additional_progress_tokens: list[ProgressToken]
    config: Config
    random_initialization: bool
    active_player: Player
    action_space: ActionSpace
    second_turn: bool = False
    terminal_state: TerminalState = TerminalState()
    incomplete: bool
    # For debugging / visualization only
    next_action: int | None = None

class PickingActionType(Enum):
    discard = "discard"
    build = "build"
    build_wonder = "build_wonder"

class PickingAction(BaseModel):
    picking_action_type: PickingActionType
    wonder: Wonder = None