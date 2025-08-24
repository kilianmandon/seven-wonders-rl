import React from "react"
import { CardColor, CardProps, PlayerState } from "./ComponentProps"

type PlayerPanelProps = {
    playerState: PlayerState
}

export function PlayerPanel({ playerState }: PlayerPanelProps) {
    const order = [
        CardColor.Brown,
        CardColor.Gray,
        CardColor.Red,
        CardColor.Green,
        CardColor.Blue,
        CardColor.Yellow,
        CardColor.Purple,
    ];

    const sortCardsByColor = (cards: CardProps[]) => {
        return [...cards].sort(
            (a, b) => order.indexOf(a.color) - order.indexOf(b.color)
        );
    };

    const staggeredCards = (cards: CardProps[]) => (
        <div className="relative h-32 w-20">
            {sortCardsByColor(cards).map((card, i) => (
                <img
                    key={card.name + i}
                    src={`/images/${card.name}_processed.png`}
                    alt={card.name}
                    className="absolute w-20 rounded-md shadow-sm"
                    style={{ top: `${i * 30}px`, left: `${i * 5}px`}}
                />
            ))}
        </div>
    );

    return (
        <div className="flex flex-col gap-4 p-3 bg-gray-100 rounded-xl shadow-lg max-w-md mx-auto">
            {/* Stats */}
            <div className="grid grid-cols-2 gap-2 text-sm font-medium">
                <div className="bg-white rounded p-1 shadow-sm">💰 {playerState.money}</div>
                <div className="bg-white rounded p-1 shadow-sm">⚔️ {playerState.military_points}</div>
                <div className="bg-white rounded p-1 shadow-sm">🏆 {playerState.victory_points}</div>
                <div className="bg-white rounded p-1 shadow-sm col-span-2">
                    🔬 {playerState.science.join(", ")}
                </div>
                <div className="bg-white rounded p-1 shadow-sm col-span-2">
                    🏴 {playerState.looted_base} | {playerState.looted_full}
                </div>
            </div>

            {/* Built Wonders */}
            <div>
                <h2 className="text-sm font-semibold mb-1">Built Wonders</h2>
                <div className="grid grid-cols-2 sm:grid-cols-2 gap-2">
                    {playerState.built_wonders.map((wonder: any, i: number) => (
                        <img
                            key={i}
                            src={`/images/wonders/${wonder.name}_processed.png`}
                            alt={wonder.name}
                            className="rounded-lg shadow w-full"
                        />
                    ))}
                </div>
            </div>

            {/* Available Wonders */}
            <div>
                <h2 className="text-sm font-semibold mb-1">Available Wonders</h2>
                <div className="grid grid-cols-2 sm:grid-cols-2 gap-2">
                    {playerState.available_wonders.map((wonder: any, i: number) => (
                        <img
                            key={i}
                            src={`/images/wonders/${wonder.name}_processed.png`}
                            alt={wonder.name}
                            className="rounded-lg shadow w-full"
                        />
                    ))}
                </div>
            </div>

            {/* Built Cards */}
            <div>
                <h2 className="text-sm font-semibold mb-1">Built Cards</h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {order.map((color) => (
                        <div key={color} className="flex flex-col items-center">
                            <span className="text-xs capitalize mb-0.5">{color}</span>
                            {staggeredCards(
                                playerState.built_cards.filter((c: CardProps) => c.color === color)
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}