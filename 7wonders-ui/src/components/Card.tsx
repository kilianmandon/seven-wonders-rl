import React from "react";
import { GameState } from "./ComponentProps";

export type CardProps = {
    name: string,
    isPlaceholder: boolean,
    onClick: ()=>void,
    clickable: boolean,
    onMouseEnter?: () => void,
    onMouseLeave?: () => void
}

export function Card({ name, isPlaceholder, onClick, clickable, onMouseEnter, onMouseLeave}: CardProps) {
    if (isPlaceholder) {
        return <div className='w-30' />
    } else {
        return <img onClick={onClick} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave} className={`w-30 brightness-120 ${clickable? "hover:scale-105 transition-transform":""}
        `} src={`/images/${name}_processed.png`}></img>
    }
}
