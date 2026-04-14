# Failed Experiments — DO NOT RETRY

## Multi-heart collection (HEART_COLLECT_TARGET=3)
- **What**: Aligners wait for 3 hearts before leaving hub
- **Result**: VOR dropped to 0.17
- **Why failed**: Deadlock — multiple aligners compete for same hearts, all waiting forever
- **Never retry because**: fundamental resource contention with shared hub

## Junction deposit at aligned junctions (off-screen)
- **What**: Miners navigate to known cogs junctions to deposit instead of hub
- **Result**: VOR 0.74 → 0.32
- **Why failed**: Junctions get scrambled by Clips during miner transit; resources lost
- **Never retry because**: scramble timing makes off-screen junction deposits unreliable
- **Note**: visible-only junction deposits (on-screen, confirmed cogs) DO work (v5)

## Phased startup (30-tick delay for agents 5-7)
- **What**: Stagger gear acquisition to reduce hub contention
- **Result**: VOR 0.71 → 0.65
- **Why failed**: 30-tick delay costs early junction captures; tick value is front-loaded
- **Never retry because**: any delay in early game directly reduces cumulative reward

## Visible junction deconfliction (strict)
- **What**: Aligners avoid junctions visible to other aligners
- **Result**: VOR 0.69 → 0.48
- **Why failed**: Too restrictive — aligners sit idle when all visible junctions are "claimed"
- **Never retry because**: observation window is only 13x13; too few junctions visible

## 4 miners / 4 aligners
- **What**: Add a 4th miner, remove 1 aligner
- **Result**: Same as 3M/5A
- **Why failed**: Extra miner doesn't offset lost aligner output
- **Never retry because**: 3 miners already saturate heart crafting pipeline

## Deposit threshold 20 (reduced from 30)
- **What**: Miners deposit earlier with less cargo
- **Result**: Minor regression
- **Why failed**: More travel trips per resource collected
- **Never retry because**: higher threshold = fewer trips = more mining time

## Aligner target timeout 15 → 10
- **What**: Reduce junction stuck threshold
- **Result**: Neutral (no improvement)
- **Why failed**: 10 ticks too aggressive — abandonsjunctions that just needed 2 more ticks
- **Status**: Not harmful, just not helpful. Could retry at 12.
