from copyreg import pickle
import itertools
import json
from fastapi import Body, FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import uvicorn

from game import discard_opponent_card, game_state_analysis, get_player_and_opponent, pick_card, pick_discarded_card, pick_progress_token, start_game, update_game_state
from game_types import ActionSpace, Age, Card, CardGroup, CardState, GameState, PickingAction, PickingActionType

app = FastAPI()

origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",  # Optional, depending on how you open the frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # You can also use ["*"] to allow all origins (not recommended in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

app.state.game_state = None

# Dummy domain 1 — GET
domain = APIRouter()

@domain.get("/reset_game")
async def get_data(request: Request, random_initialization: bool = True):
    request.app.state.game_state = start_game(random_initialization)
    return { "state": request.app.state.game_state }

@domain.post("/set_guild_labels")
async def set_guild_labels(request: Request, guild_labels: list[bool]=Body(...)):
    state: GameState = request.app.state.game_state
    if state.age == Age.age_three and state.incomplete:
        assert len(a for a in guild_labels if a)==3, 'Invalid number of guild labels.'
        for i, label in enumerate(guild_labels):
            state.cards[i].group = CardGroup.guild if label else CardGroup.age_three
    else:
        raise RuntimeError('Guild labeling not necessary in this stage.')
    
    return { "state": request.app.state.game_state }

@domain.post("/set_card_names")
async def set_card_names(request: Request, card_names: list[str]=Body(...)):
    state: GameState = request.app.state.game_state
    if state.incomplete:
        for i, card_name in enumerate(card_names):
            if card_name != state.cards[i].name:
                new_card = next(a for a in state.config.cards[state.cards[i].group.value] if a.name==card_name)
                state.cards[i] = new_card

        update_game_state(state)
            
    else:
        raise RuntimeError('State is complete, no card labels required.')
    return { "state": request.app.state.game_state }

@domain.get("/pick_card")
async def pick_card_route(request: Request, card_idx: int, picking_type: int):
    state: GameState = request.app.state.game_state
    action_mask = np.array(state.action_space.action_mask).reshape(20, 14)
    if not action_mask[card_idx][picking_type]:
        raise RuntimeError('Invalid action selected.')

    player, opponent = get_player_and_opponent(state)
    if picking_type==0:
        action_type = PickingAction(picking_action_type = PickingActionType.discard)
    elif picking_type==1:
        action_type = PickingAction(picking_action_type=PickingActionType.build)
    else:
        action_type = PickingAction(picking_action_type=PickingActionType.build_wonder, wonder=state.config.wonders[picking_type-2])

    pick_card(state, player, opponent, card_idx, action_type)

    return { "state": state }

@domain.get("/pick_progress_token")
async def pick_progress_token_route(request: Request, token_name: str):
    state: GameState = request.app.state.game_state
    player, opponent = get_player_and_opponent(state)
    idx = next(i for i, c in enumerate(state.config.all_progress_tokens) if c.value==token_name)
    pick_progress_token(state, player, idx)

    return {"state": state}

@domain.get("/discard_opponent_card")
async def discard_opponent_card_route(request: Request, card_name: str):
    state: GameState = request.app.state.game_state
    player, opponent = get_player_and_opponent(state)
    all_cards = list(itertools.chain(*state.config.cards))
    card: Card = next(c for c in all_cards if c.name==card_name)
    all_color_cards = [c for c in all_cards if c.color==card.color]
    idx = all_color_cards.index(card)

    discard_opponent_card(state, player, opponent, card.color, idx)

    return {"state": state}

@domain.get("/pick_discarded_card")
async def pick_discarded_card_route(request: Request, card_name: str):
    state: GameState = request.app.state.game_state
    player, opponent = get_player_and_opponent(state)

    all_cards = list(itertools.chain(*state.config.cards))
    card: Card = next(c.name==card_name for c in all_cards)
    idx = all_cards.index(card)

    pick_discarded_card(state, player, opponent, idx)

    return {"state": state}

@domain.get("/change_replay_frame")
async def change_replay_frame(request: Request, offset: int):
    states = request.app.state.replay['states']
    current_frame = request.app.state.replay['current_frame']
    new_frame = max(0, min(len(states)-1, current_frame+offset))

    request.app.state.replay['current_frame'] = new_frame
    if new_frame < len(request.app.state.replay['actions']):
        states[new_frame].next_action = request.app.state.replay['actions'][new_frame]

    request.app.state.game_state = states[new_frame]

    return { "state": states[new_frame] }

@domain.get("/load_replay")
async def load_replay(request: Request, replay_name: str):
    with open(f'replays/{replay_name}.json') as f:
        data = json.load(f)

    states = [GameState.model_validate(state_data) for state_data in data['states']]
    for state in states:
        game_state_analysis(state)

    actions = data['actions']
    request.app.state.replay = {
        'states': states,
        'actions': actions,
        'current_frame': 0,
    }

    request.app.state.game_state = states[0]

    return {"state": states[0]}


app.include_router(domain, prefix="")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)