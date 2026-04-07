# Classic Overcogged

A fully cooperative cog assembly game. Cog agents work together to gather
elements, convert them into hearts at hubs, and deliver hearts to junctions.

## Core Loop

1. Start with miner gear equipped.
2. Pick up carbon from a carbon extractor.
3. Deposit carbon into a hub. Three deposits are required.
4. Wait for the hub to finish converting.
5. Swap to scrambler gear at a scrambler station.
6. Pick up the heart from the hub.
7. Deliver the heart to the junction.

## Interaction Model

All stations and objects are solid. Agents interact by walking into them. The
interaction fires while the agent stays in its current cell.

## Gear System

- Gear slot: holds either miner or scrambler.
- Carry slot: holds either an element or a heart.

Miner enables element pickup. Scrambler enables heart pickup. Agents start with
miner gear and swap at the corresponding station.

## Objects

- Carbon extractor: infinite carbon dispenser for miners with an empty carry slot.
- Hub: accepts ingredients, cooks automatically, and yields a heart for a scrambler.
- Miner station: equips miner gear.
- Scrambler station: equips scrambler gear.
- Chest: temporary storage for passing items between agents.
- Junction: delivery point that consumes hearts for reward.

## Variants

- `recipes`: alternate elements and per-hub recipe schedules.
- `tips`: bonus delivery reward.
- `burn`: hearts burn if left in hubs too long.
- `short_cook`: halve cook time.
- `long_cook`: double cook time.
- `fast_burn`: halve burn time.
- `cramped_kitchen`: smaller map with tighter corridors.
- `full`: combine `recipes` and `burn`.
