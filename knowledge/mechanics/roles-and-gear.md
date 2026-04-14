# Roles & Gear

Source: `machina_1.py:101-106`, `roles/*.py`

## Gear Costs (deducted from hub inventory)

| Role | Carbon | Oxygen | Germanium | Silicon | Total |
|------|--------|--------|-----------|---------|-------|
| Miner | 1 | 1 | **3** | 1 | 6 |
| Aligner | **3** | 1 | 1 | 1 | 6 |
| Scrambler | 1 | **3** | 1 | 1 | 6 |
| Scout | 1 | 1 | 1 | **3** | 6 |

## Gear Bonuses

| Role | Bonus | Purpose |
|------|-------|---------|
| Miner | +40 cargo, +10 extract/interaction | Resource gathering |
| Aligner | Enables alignment action | Junction capture |
| Scrambler | +200 HP (total 300) | Enemy territory raids |
| Scout | +400 HP, +100 energy (total 500 HP, 120 energy) | Exploration |

## Hub Starting Inventory
- 24 each of: carbon, oxygen, germanium, silicon (96 total)
- 5 pre-crafted hearts
- Heart craft cost: 7 each element = 28 total

### Initial Budget
- 8 agents × 6 resources = 48 for full gear (half of starting inventory)
- 5 hearts ready immediately
- ~3 more hearts from remaining resources (24-gear_costs)/7 per element
- First wave: 5 aligners can get hearts at tick 0

## Gear Destruction
- Destroyed on agent death (HP reaches 0)
- Must re-acquire at gear station (costs hub resources again)
- Protecting geared agents is resource-efficient

## Station Interaction
- Auto-triggered when agent at team's gear station
- Deducts cost from hub inventory (not agent inventory)
- Agent receives gear item (limit: 1 gear per agent)
