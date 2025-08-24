import { useEffect, useState } from 'react'
import React from 'react'
import { CardProps } from './components/Card';
import { CardGrid } from './components/CardGrid';
import { ActionType, Age, CardColor, CardGroup, CardState, GameState, Player, ProgressToken } from './components/ComponentProps';
import { ActionSelection, ActionSelectionProps, availableCards, ColourOption } from './components/ActionSelection';
import { PlayerPanel } from './components/PlayerPanel';

type GuildLabelingProps = {
  gameState: GameState
  onCompletion: (flags: boolean[]) => void

}

function GuildLabeling({ gameState, onCompletion }: GuildLabelingProps) {
  const [guildLabels, setGuildLabels] = useState<boolean[]>(Array(20).fill(false));
  const [cardProps, setCardProps] = useState<CardProps[]>([]);

  const setCardPropsForGuildLabeling = () => {
    if (gameState != null) {
      let cardProps: CardProps[] = gameState?.cards.map((c, i) => {
        let name = guildLabels[i] ? 'hidden_guilds' : 'missing';
        let isPlaceholder = false;
        let onClick = () => {
          setGuildLabels(labels => labels.map((l, j) => i != j ? l : !l));
        };
        let clickable = true;

        return { name, isPlaceholder, onClick, clickable };
      });
      setCardProps(cardProps);
    }
  }




  useEffect(setCardPropsForGuildLabeling, guildLabels);

  return <>
    <CardGrid cards={cardProps} age={gameState.age} />
    {guildLabels.filter(x => x).length == 3 &&
      <button onClick={() => onCompletion(guildLabels)}>Continue</button>
    }
  </>
}

type CardLabelingProps = {
  gameState: GameState,
  onCompletion: (names: string[]) => void,
  setActionSelectionProps: (props: ActionSelectionProps) => void
}

function MissingCardLabeling({ gameState, onCompletion, setActionSelectionProps }: CardLabelingProps) {
  const [cardNames, setCardNames] = useState<string[]>(gameState.cards.map(c => c.name));
  const [cardProps, setCardProps] = useState<CardProps[]>([]);

  const setCardPropsForCardLabeling = () => {
    let cardProps = gameState.cards.map((c, i) => {
      let name = cardNames[i];
      let clickable = false;
      let isPlaceholder = false;
      let onClick = () => { };
      if (gameState.cards_mask[i] == CardState.Hidden) {
        if (c.group == CardGroup.Age_One) {
          name = "hidden_age_1";
        } else if (c.group == CardGroup.Age_Two) {
          name = "hidden_age_2";
        } else if (c.group == CardGroup.Age_Three) {
          name = "hidden_age_3";
        } else if (c.group == CardGroup.Guild) {
          name = "hidden_guilds";
        }
      }
      else if (name == "") {
        name = "missing";
        clickable = true;
        onClick = () => {
          let actionSelectionProps = {
            options: availableCards(gameState, i, cardNames),
            callback: (newCardName: string) => {
              console.log(`Card ${newCardName} selected!`);
              setCardNames(oldCardNames => oldCardNames.map((oldCardName, j) => i == j ? newCardName : oldCardName));
            }
          }
          console.log(actionSelectionProps);
          setActionSelectionProps(actionSelectionProps);
        }
      }
      return { name, clickable, onClick, isPlaceholder };

    });
    setCardProps(cardProps);
    if (cardNames.every((x, i) => x != "" || gameState.cards_mask[i] != CardState.Revealed)) {
      console.log("Every card is labeled!");
      onCompletion(cardNames);
    }
  }

  useEffect(setCardPropsForCardLabeling, cardNames);
  return <>
    <CardGrid cards={cardProps} age={gameState.age} />
  </>
}

type CardPickingProps = {
  gameState: GameState,
  onCompletion: (cardIdx: number, pickingTypeIdx: number) => void,
  setActionSelectionProps: (props: ActionSelectionProps) => void

};

function suggestedAction(gameState: GameState) {
  let actionScore = gameState.action_space.action_score;
  let actionMask = gameState.action_space.action_mask;

  let vmax = 0;
  let maxIdx = -1;
  actionScore.forEach((v, i) => {
      if (v > vmax && actionMask[i]) {
        vmax = v;
        maxIdx = i;
      }
  });

  if (gameState.action_space.action_type==ActionType.PickCard) {
    let cardIdx = Math.floor(maxIdx / 14);
    let actionIdx = maxIdx % 14;
    if (actionIdx == 0) {
      return `Sell ${gameState.cards[cardIdx].name}`
    } else if (actionIdx==1) {
      return `Build ${gameState.cards[cardIdx].name}`
    } else {
      let wonder = gameState.config.wonders[actionIdx-2];
      return `Build Wonder ${wonder} with ${gameState.cards[cardIdx].name}`
    }
  } else return "Special Action";
}

function reshape1Dto2D<T>(arr: T[], cols: number): T[][] {
  if (cols <= 0) throw new Error("Number of columns must be greater than 0");
  const result: T[][] = [];
  for (let i = 0; i < arr.length; i += cols) {
    result.push(arr.slice(i, i + cols));
  }
  return result;
}

function CardPicking({ gameState, onCompletion, setActionSelectionProps }: CardPickingProps) {
  const [cardProps, setCardProps] = useState<CardProps[]>([]);
  const [hoveredCard, setHoveredCard] = useState<number | undefined>(undefined);
  let actionMask = reshape1Dto2D(gameState.action_space.action_mask, 14);
  let actionCost = reshape1Dto2D(gameState.action_space.action_cost, 14);
  let actionScore = reshape1Dto2D(gameState.action_space.action_score, 14);

  function maskedMax(mask, values) {
    const maskedVals = mask.flatMap((row, i) =>
      row.map((m, j) => (m ? values[i][j] : null))
    ).filter(v => v !== null);
  
    return maskedVals.length ? Math.max(...maskedVals) : -1e5;
  }



  let maxScore = maskedMax(actionMask, actionScore);

  let actionScoreColors = actionScore.map(scoreList => scoreList.map(score => {
    if (score/maxScore>0.95) return "green"
    else if (score/maxScore>0.8) return "yellow"
    else if (score/maxScore>0.6) return "orange"
    else return "red"
  }));


  const setCardPropsForCardLabeling = () => {
    let cardProps = gameState.cards.map((c, i) => {
      let clickable = actionMask[i].some(x => x);
      let name = c.name;

      if (gameState.cards_mask[i] == CardState.Hidden) {
        if (c.group == CardGroup.Age_One) {
          name = "hidden_age_1";
        } else if (c.group == CardGroup.Age_Two) {
          name = "hidden_age_2";
        } else if (c.group == CardGroup.Age_Three) {
          name = "hidden_age_3";
        } else if (c.group == CardGroup.Guild) {
          name = "hidden_guilds";
        }
      }

      let isPlaceholder = gameState.cards_mask[i] == CardState.Taken;
      let onClick = () => { };
      let onMouseEnter = ()=>{};
      let onMouseLeave = ()=>{};
      if (clickable) {
        let options: ColourOption[] = [];
        if (actionMask[i][0]) {
          options.push({ value: "0", label: `Sell (${actionCost[i][0]} coins) [${actionScore[i][0]}]`, color: actionScoreColors[i][0]});
        } if (actionMask[i][1]) {
          options.push({ value: "1", label: `Build (${actionCost[i][1]} coins) [${actionScore[i][1]}]`, color: actionScoreColors[i][1] });
        }
        actionMask[i].slice(2).forEach((exists, j) => {
          if (exists) {
            options.push({ value: `${j + 2}`, label: `Build Wonder ${gameState.config.wonders[j].name} (${actionCost[i][j + 2]} coins) [${actionScore[i][j+2]}]`, color: actionScoreColors[i][j+2] });
          }
        });
        onMouseEnter = ()=> setHoveredCard(i);
        onMouseLeave = ()=> setHoveredCard(undefined);

        onClick = () => setActionSelectionProps({
          options: options, callback: (value) => {
            console.log(`Picked Card: ${i} | ${value}`);
            onCompletion(i, parseInt(value));
          }
        })

      }
      return { name, clickable, onClick, isPlaceholder, onMouseEnter, onMouseLeave };

    });
    setCardProps(cardProps);
  }

  useEffect(setCardPropsForCardLabeling, [gameState]);
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (hoveredCard === undefined) { console.log("No card hovered"); return;}
      if (!cardProps[hoveredCard]?.clickable) { console.log("No clickable card hovered."); return;}

      if (e.key.toLowerCase() === "b" && actionMask[hoveredCard][1]) {
        // Build (standard)
        onCompletion(hoveredCard, 1);
      } else if (e.key.toLowerCase() === "s" && actionMask[hoveredCard][0]) {
        // Sell
        onCompletion(hoveredCard, 0);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [hoveredCard, cardProps, actionMask, onCompletion]);

  let player = gameState.player_1_state;
  if (gameState.active_player == Player.Two) {
    player = gameState.player_2_state;
  }
  return <>
    <h2>Pick a Card ({player.money} coins)</h2>
    <CardGrid cards={cardProps} age={gameState.age} />
  </>;
}

type ProgressTokenPickingProps = {
  gameState: GameState,
  onCompletion: (name: string) => void,
  setActionSelectionProps: (props: ActionSelectionProps) => void

}
function PickProgressToken({ gameState, onCompletion, setActionSelectionProps }: ProgressTokenPickingProps) {
  useEffect(() => {
    const options = gameState.config.all_progress_tokens.filter((token, i) => gameState.action_space.action_mask[i]).map(token => {
      return {
        value: token, label: token, color: "black"
      };
    });
    const callback = (value) => onCompletion(value);
    setActionSelectionProps({ options, callback });
  }, [gameState]);
  return <h2>Pick a Progress Token</h2>;
}

type DiscardOpponentCardProps = {
  gameState: GameState,
  cardColor: CardColor,
  onCompletion: (name: string) => void,
  setActionSelectionProps: (props: ActionSelectionProps) => void

}
function DiscardOpponentCard({ gameState, cardColor, onCompletion, setActionSelectionProps }: DiscardOpponentCardProps) {
  useEffect(() => {
    const colorCards = gameState.config.cards.flat().filter(c => c.color == cardColor);
    const options = colorCards.filter((card, i) => gameState.action_space.action_mask[i]).map(card => {
      return {
        value: card.name, label: card.name, color: "black"
      };
    });
    const callback = (value) => onCompletion(value);
    setActionSelectionProps({ options, callback });
  }, [gameState]);
  return <h2>Discard opponent card</h2>;
}

type PickDiscardedProgressTokenProps = {
  gameState: GameState,
  onCompletion: (name: string) => void,
  setActionSelectionProps: (props: ActionSelectionProps) => void

}
function PickDiscardedProgressToken({ gameState, onCompletion, setActionSelectionProps }: PickDiscardedProgressTokenProps) {
  useEffect(() => {
    const options = gameState.config.all_progress_tokens.filter((token, i) => gameState.action_space.action_mask[i]).map(token => {
      return {
        value: token, label: token, color: "black"
      };
    });
    const callback = (value) => onCompletion(value);
    setActionSelectionProps({ options, callback });
  }, [gameState]);
  return <h2>Pick discarded progress token</h2>;
}

type PickDiscardedCardProps = {
  gameState: GameState,
  onCompletion: (name: string) => void,
  setActionSelectionProps: (props: ActionSelectionProps) => void

}
function PickDiscardedCard({ gameState, onCompletion, setActionSelectionProps }: PickDiscardedProgressTokenProps) {
  useEffect(() => {
    const options = gameState.config.cards.flat().filter((card, i) => gameState.action_space.action_mask[i]).map(card => {
      return { value: card.name, label: card.name, color: "black"}
    });

    const callback = (value) => onCompletion(value);
    setActionSelectionProps({ options, callback });
  }, [gameState]);
  return <h2>Pick discarded card</h2>;
}

type TerminalStateProps = {
  gameState: GameState
}

function TerminalStateDisplay({gameState} : TerminalStateProps) {
  let terminalState = gameState.terminal_state;
  return <div className="flex flex-col">
  {terminalState.draw ? <text>Draw</text> : 
  <>
  <text>Player {terminalState.winner} won</text>
  <text>Victory: {terminalState.victory_type}</text>
  </>
  }
  </div>
}


function App() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [cardProps, setCardProps] = useState<CardProps[]>([]);
  const [actionSelectionProps, setActionSelectionProps] = useState<ActionSelectionProps | null>(null);



  const resetGame = () => {
    fetch('http://localhost:8000/reset_game?random_initialization=true').then(res => res.json()).then(
      data => {
        let gs = data['state'] as GameState;
        console.log(gs);
        setGameState(data['state'] as GameState)
      }
    )
  }
  const loadReplay = () => {
    fetch('http://localhost:8000/load_replay?replay_name=replay_1').then(res => res.json()).then(
      data => {
        let gs = data['state'] as GameState;
        console.log(gs);
        setGameState(data['state'] as GameState)
      }
    )
  }
  const changeReplayFrame = (i: number) => {
    fetch(`http://localhost:8000/change_replay_frame?offset=${i}`).then(res => res.json()).then(
      data => {
        let gs = data['state'] as GameState;
        console.log(gs);
        setGameState(data['state'] as GameState)
      }
    )
  }

  const sendGuildLabels = (flags: boolean[]) => {
    fetch('http://localhost:8000/set_guild_labels', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(flags),
    }).then(res => res.json()).then(data =>
      setGameState(data['state'] as GameState)
    );
  }
  const sendCardNameLabels = (cardNames: string[]) => {
    fetch('http://localhost:8000/set_card_names', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(cardNames),
    }).then(res => res.json()).then(data =>
      setGameState(data['state'] as GameState)
    );
  }

  const sendPickCard = (cardIdx: number, pickingTypeIdx: number) => {
    fetch(`http://localhost:8000/pick_card?card_idx=${cardIdx}&picking_type=${pickingTypeIdx}`).then(res => res.json()).then(data => { console.log(data.state as GameState); setGameState(data.state as GameState); }
    )
  };

  const sendPickProgressToken = (tokenName: string) => {
    fetch(`http://localhost:8000/pick_progress_token?token_name=${tokenName}`).then(res => res.json()).then(data => { console.log(data.state as GameState); setGameState(data.state as GameState); }
    )
  };

  const sendDiscardOpponentCard = (cardName: string) => {
    fetch(`http://localhost:8000/discard_opponent_card?card_name=${cardName}`).then(res => res.json()).then(data => { console.log(data.state as GameState); setGameState(data.state as GameState); }
    )
  };
  const sendPickDiscardedCard = (cardName: string) => {
    fetch(`http://localhost:8000/pick_discarded_card?card_name=${cardName}`).then(res => res.json()).then(data => { console.log(data.state as GameState); setGameState(data.state as GameState); }
    )
  };


  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") {
        changeReplayFrame(1);
      } else if (e.key === "ArrowLeft") {
        changeReplayFrame(-1);
      } else if (e.key === "Escape") {
        setActionSelectionProps(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });






  const setActionSelectionPropsWithClosing = (arg: ActionSelectionProps) => {
    let newProps = { ...arg };
    const originalCallback = arg.callback;
    newProps.callback = (value: string) => {
      originalCallback(value);
      setActionSelectionProps(null);
    }
    setActionSelectionProps(newProps);
  }

  const Board = (gameState: GameState) => {
    console.log("Rebuilding board.");
    const terminal = gameState.terminal_state.is_terminal;
    const guildLabeling = !terminal && gameState.incomplete && (gameState.age == Age.Three && gameState.cards.filter(c => c.group == CardGroup.Guild).length != 3);
    const missingLabeling = !terminal && gameState.incomplete && !guildLabeling;
    const picking = !terminal && !gameState.incomplete && gameState.action_space.action_type == ActionType.PickCard;
    const pickingProgressToken = !terminal && !gameState.incomplete && gameState.action_space.action_type==ActionType.PickProgressToken;
    const discardingGray = !terminal && !gameState.incomplete && gameState.action_space.action_type==ActionType.DiscardOpponentGray;
    const discardingBrown = !terminal && !gameState.incomplete && gameState.action_space.action_type==ActionType.DiscardOpponentBrown;
    const pickingDiscardedProgressToken = !terminal && !gameState.incomplete && gameState.action_space.action_type==ActionType.PickDiscardedProgressToken;
    const pickingDiscardedCard = !terminal && !gameState.incomplete && gameState.action_space.action_type==ActionType.PickDiscardedCard;

    const other = ![terminal, guildLabeling, missingLabeling, picking, pickingProgressToken, discardingGray, discardingBrown, pickingDiscardedProgressToken, pickingDiscardedCard].some(x=>x);

    return <>
    <div className="flex flex-col items-center gap-3">
      <h1>{`Player ${gameState.active_player}`}</h1>
      <div className="min-h-[80vh]">
      {terminal && <TerminalStateDisplay gameState={gameState}/>}
      {guildLabeling && <GuildLabeling gameState={gameState} onCompletion={sendGuildLabels} />}
      {missingLabeling && <MissingCardLabeling gameState={gameState} onCompletion={sendCardNameLabels} setActionSelectionProps={setActionSelectionPropsWithClosing} />}
      {picking && <CardPicking gameState={gameState} onCompletion={sendPickCard} setActionSelectionProps={setActionSelectionPropsWithClosing} />}
      {pickingProgressToken && <PickProgressToken gameState={gameState} onCompletion={sendPickProgressToken} setActionSelectionProps={setActionSelectionPropsWithClosing}/>}
      {discardingGray && <DiscardOpponentCard cardColor={CardColor.Gray} gameState={gameState} onCompletion={sendDiscardOpponentCard} setActionSelectionProps={setActionSelectionPropsWithClosing}/>}
      {discardingBrown && <DiscardOpponentCard cardColor={CardColor.Brown} gameState={gameState} onCompletion={sendDiscardOpponentCard} setActionSelectionProps={setActionSelectionPropsWithClosing}/>}
      {pickingDiscardedProgressToken && <PickDiscardedProgressToken gameState={gameState} onCompletion={sendPickProgressToken} setActionSelectionProps={setActionSelectionPropsWithClosing}/>}
      {pickingDiscardedCard && <PickDiscardedCard gameState={gameState} onCompletion={sendPickDiscardedCard} setActionSelectionProps={setActionSelectionPropsWithClosing}/>}
      {other && <h1>"Undefined State!"</h1>}
      </div>
      <h2>{gameState.active_player===Player.One && `Suggested ${suggestedAction(gameState)}`}</h2>
      </div>
    </>
  };

  return (
    <>
      {gameState != null ?
        <>
          {actionSelectionProps != null && <ActionSelection {...actionSelectionProps} />}
          <div className="flex flex-col">
          <div className="flex flex-row justify-between">
            <PlayerPanel playerState={gameState.player_1_state}/>
            {Board(gameState)}
            <PlayerPanel playerState={gameState.player_2_state}/>
          </div>
          <div className="flex flex-row">
            <button onClick={()=>changeReplayFrame(-1)}>Previous Frame</button>
            <button onClick={()=>changeReplayFrame(1)}>Next Frame</button>
          </div>
          </div>
        </> :
        <>
        <button onClick={resetGame}>Start Game</button>
        <button onClick={loadReplay}>Load Replay</button>
        </>
      }
    </>
  )
}

export default App
