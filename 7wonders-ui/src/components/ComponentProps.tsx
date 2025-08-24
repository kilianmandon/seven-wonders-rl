// Enums
export enum PlayerScope {
    Self = "self",
    Any = "any",
}

export enum CardColor {
    Brown = "brown",
    Gray = "gray",
    Red = "red",
    Green = "green",
    Blue = "blue",
    Yellow = "yellow",
    Purple = "purple",
}

export enum CardGroup {
    Age_One = 0,
    Age_Two = 1,
    Age_Three = 2,
    Guild = 3
}

export enum SpecialColors {
    Wonder = "wonder",
    ThreeCoins = "three_coins",
}

export enum ResourceType {
    Wood = "wood",
    Clay = "clay",
    Stone = "stone",
    Glass = "glass",
    Papyrus = "papyrus",
}

export enum ScienceType {
    Astrolabe = "astrolabe",
    Scales = "scales",
    SolarClock = "solar_clock",
    Mortar = "mortar",
    PlumbTriangle = "plumb_triangle",
    Quill = "quill",
    Wheel = "wheel",
}

export enum ProgressToken {
    Agriculture = "agriculture",
    Architecture = "architecture",
    Economy = "economy",
    Law = "law",
    Masonry = "masonry",
    Mathematics = "mathematics",
    Philosophy = "philosophy",
    Strategy = "strategy",
    Theology = "theology",
    Urbanism = "urbanism",
}

export enum WonderActions {
    DiscardGray = "discard_gray",
    RandomProgressToken = "random_progress_token",
    ChooseDiscarded = "choose_discarded",
    DiscardBrown = "discard_brown",
}

export enum Age {
    One = 0,
    Two = 1,
    Three = 2,
}

export enum CardState {
    Taken = "taken",
    Revealed = "revealed",
    Hidden = "hidden",
    HiddenGuild = "hidden_guild",
}

export enum PickActionType {
    Discard = "discard",
    Build = "build",
    BuildWonder = "build_wonder",
}

export enum Player {
    One = 0,
    Two = 1
}

export enum ActionType {
    NoActionRequired = 0,
    PickCard = 1,
    PickProgressToken = 2,
    DiscardOpponentGray = 3,
    DiscardOpponentBrown = 4,
    PickDiscardedProgressToken = 5,
    PickDiscardedCard = 6
}

export interface ActionSpace {
    action_type: ActionType;
    action_mask: boolean[];
    action_cost: number[];
    action_score: number[];
}

// Interfaces
export interface WonderProps {
    name: string;
    cost_resources: ResourceType[];
    resources_provided: (ResourceType | ResourceType[])[];
    military_points_provided: number;
    victory_points_provided: number;
    money_provided: number;
    money_opponent_lost: number;
    second_turn: boolean;
    action: WonderActions | null;
}

export interface CardProps {
    name: string;
    group: CardGroup;
    color: CardColor;
    cost_resources: ResourceType[];
    cost_money: number;
    resources_provided: (ResourceType | ResourceType[])[];
    military_points_provided: number;
    victory_points_provided: number;
    science_provided: ScienceType | null;
    deals_provided: ResourceType[];
    money_provided: number;
    money_per_building: [number, CardColor | [CardColor, CardColor], PlayerScope] | null;
    victory_points_per_building: [number, CardColor | SpecialColors, PlayerScope] | null;
}

export interface Config {
    age_orders: number[][][];
    wonders: WonderProps[];
    cards: CardProps[][];
    transitions: [CardProps, CardProps][];
    all_progress_tokens: ProgressToken[];
}

export interface PlayerState {
    money: number;
    victory_points: number;
    military_points: number;
    science: ScienceType[];
    progress_tokens: ProgressToken[];

    available_wonders: WonderProps[];
    built_wonders: WonderProps[];
    built_cards: CardProps[];
    looted_base: boolean;
    looted_full: boolean;
}

export interface TerminalState {
    is_terminal: boolean;
    victory_type: string;
    draw: boolean;
    winner: Player;
}

export interface GameState {
    player_1_state: PlayerState;
    player_2_state: PlayerState;
    age: Age;
    cards: CardProps[];
    cards_mask: CardState[];
    discarded_cards: CardProps[];
    config: Config;
    random_initialization: boolean;
    active_player: Player;
    action_space: ActionSpace;
    incomplete: boolean;
    terminal_state: TerminalState;
}