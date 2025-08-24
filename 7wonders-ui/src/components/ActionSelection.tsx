import React from "react";
import Select, { SingleValue, MultiValue, StylesConfig } from "react-select";
import { CardGroup, CardState, GameState } from "./ComponentProps";
import chroma from "chroma-js";

export const availableCards = (gameState: GameState, cardIndex: number, openCardNames: string[]) => {

    let availableCards = gameState.config.cards[gameState.age].slice();
    let availableGuildCards = gameState.config.cards[3].slice();
    openCardNames.forEach((cardName, i) => {
        if (gameState.cards_mask[i] == CardState.Revealed) {
            availableCards = availableCards.filter(x => x.name != cardName);
            availableGuildCards = availableGuildCards.filter(x => x.name != cardName);
        }
    });
    gameState.discarded_cards.forEach(c => {
        availableCards = availableCards.filter(x => x.name != c.name);
        availableGuildCards = availableGuildCards.filter(x => x.name != c.name);
    });


    if (gameState.cards[cardIndex].group == CardGroup.Guild) {
        return availableGuildCards.map(c => {
            return { value: c.name, label: c.name, color: "black" }
        }
        );
    } else {
        return availableCards.map(c => {
            return { value: c.name, label: c.name, color: "black" }
        }
        );
    }
};

export type ColourOption = {
    value: string,
    label: string,
    color: string
};

// export type Option = {
//     value: string,
//     label: string
// };

export type ActionSelectionProps = {
    options: ColourOption[],
    callback: (label: string) => void
};
export const ActionSelection = ({ options, callback }: ActionSelectionProps) => {
    const handleChange = (option: SingleValue<ColourOption>) => {
        if (option) {
          callback(option.value);
        }
      };

      const colourStyles: StylesConfig<ColourOption> = {
        control: (styles) => ({ ...styles, backgroundColor: 'white' }),
        option: (styles, { data, isDisabled, isFocused, isSelected }) => {
          const color = chroma(data.color);
          return {
            ...styles,
            backgroundColor: isDisabled
              ? undefined
              : isSelected
              ? data.color
              : isFocused
              ? color.alpha(0.1).css()
              : undefined,
            color: isDisabled
              ? '#ccc'
              : isSelected
              ? chroma.contrast(color, 'white') > 2
                ? 'white'
                : 'black'
              : data.color,
            cursor: isDisabled ? 'not-allowed' : 'default',
      
            ':active': {
              ...styles[':active'],
              backgroundColor: !isDisabled
                ? isSelected
                  ? data.color
                  : color.alpha(0.3).css()
                : undefined,
            },
          };
        },
      };

    return <div className="fixed top-10 left-1/2 -translate-x-1/2 z-50 w-64 p-4 bg-white rounded-xl shadow-2xl ring-1 ring-black/10 backdrop-blur-md">
        <Select styles={colourStyles} options={options} isMulti={false} isSearchable onChange={handleChange} autoFocus menuIsOpen={true}/>
        </div>;

}