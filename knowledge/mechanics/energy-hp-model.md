# Energy & HP Model

## Energy
Source: `energy.py`, `solar.py`, `days.py`

| Parameter | Value |
|-----------|-------|
| Base limit | 20 |
| Initial | 100 |
| Move cost | 4 per move |
| Territory regen | +100/tick (full restore) |

### Solar (Day/Night Cycle)
- Day length: 200 ticks
- Day solar: 3/tick (ticks 0-99, 200-299, 400-499, ...)
- Night solar: 1/tick (ticks 100-199, 300-399, 500-599, ...)

### Movement Budget
| Period | Solar | Moves per tick | Effective speed |
|--------|-------|---------------|----------------|
| Day (outside territory) | 3/tick | ~0.75/tick | Fast |
| Night (outside territory) | 1/tick | ~0.25/tick | **4x slower** |
| Inside territory | +100/tick | unlimited | Full speed always |

**Night is devastating for agents outside territory.** Route through friendly zones during night.

## HP
Source: `damage.py`, `heal_team.py`

| Parameter | Value |
|-----------|-------|
| Max | 100 |
| Initial | 50 |
| Drain | -1/tick (everywhere) |
| Territory heal | +100/tick (net +99 inside) |

### Survival Window
- Starting HP 50, no territory: dies in 50 ticks
- With scrambler gear (+200 HP): 300 HP total, survives 300 ticks outside territory
- With scout gear (+400 HP): 500 HP total, survives 500 ticks outside territory

### Death Consequences
- ALL hearts destroyed
- ALL gear destroyed
- Must re-acquire at stations (costs hub resources again)

## Territory Radius
| Source | Radius | Coverage |
|--------|--------|----------|
| Hub | 20 cells | ~1256 cells |
| Junction | 10 cells | ~314 cells |

Overlapping territories: strongest team wins cell.
