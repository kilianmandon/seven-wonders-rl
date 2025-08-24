import React from "react";
import { Card, CardProps } from "./Card";


export function CardGrid({ cards, age }: { cards: CardProps[], age: number }) {
    const rowLayouts = [
        [2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2],
        [2, 3, 4, 2, 4, 3, 2]
    ];

    const rows = rowLayouts[age];

    const splitCards: CardProps[][] = [];
    let j = 0;
    rows.forEach(i => {
        let splitRow: CardProps[] = [];
        for (let x of [...Array(i).keys()]) {
            splitRow.push(cards[j]);
            j += 1;
        }
        splitCards.push(splitRow);
    });
    if (age==2)
        splitCards[3].splice(1, 0, { name: '', isPlaceholder: true, onClick: ()=>{}, clickable: false });

    return <>
        <div className='flex flex-col items-center'>
            {
                splitCards.map((cardPropsList, i) =>
                    <div className={`flex flex-row gap-4 z-${i} ${i > 0 ? '-mt-20' : ''}`}>
                        {cardPropsList.map(cardProps => {
                            return <Card {...cardProps} />
                        })
                        }
                    </div>
                )

            }
        </div>
    </>

}