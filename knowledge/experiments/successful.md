# Successful Changes

## Hub inventory awareness (v4)
- **What**: Parse `team:*` global tokens to track hub resources. Aligners check if hub can craft hearts before going.
- **Impact**: Local VOR 0.30 → 0.65 (+117%)
- **Why it works**: Aligners no longer camp at hub waiting for hearts that can't be crafted. They explore and discover junctions instead.
- **Lines**: `SoftyState.hub_resources`, aligner `_aligner()` hub_can_craft check

## `last_action_move` stuck detection (v4)
- **What**: Parse `last_action_move` token. Failed move (+2 stuck). Reduced severe threshold from 8 to 5.
- **Impact**: Contributed to 0.30 → 0.65 jump
- **Why it works**: Agents hitting walls or blocked by others recover faster

## 3M/5A role split (v4)
- **What**: Remove scramblers and scout. 3 miners sustain economy, 5 aligners maximize captures.
- **Impact**: Better than 3M/2S/2A/1Sc baseline
- **Why it works**: Pure alignment at 1 heart/junction beats scramble+align at 2 hearts/junction
- **Caveat**: This was BEFORE discovering cascade failure. May need re-evaluation with scramblers.

## Frontier-aware junction targeting (v4)
- **What**: Score junctions by `frontier * 8.0 - distance` where frontier = neutral junctions that become alignable
- **Impact**: Marginal improvement, kept for directional correctness
- **Why it works**: Prioritizes junctions that unlock new territory

## Visible-only junction deposits (v5)
- **What**: Miners deposit only at junctions visible on screen (confirmed cogs), fall back to hub for off-screen
- **Impact**: Local 8v0 score 1.11 → 1.27
- **Why it works**: Eliminates risk of depositing at scrambled junctions
