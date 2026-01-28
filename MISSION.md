# MISSION BRIEFING: CogsGuard Deployment

## Territorial Defense Initiative

Welcome, Cognitive!

You are part of an elite unit deployed to capture and defend territory against hostile Clips. Your mission: gather
resources, acquire specialized gear, and work with your team to control junctions. Territory control is everything. Do
not let the Clips overrun the sector.

This report arms you with the latest intelligence on what you might face. In different environments, specific
**details** may vary.

---

## YOUR LOADOUT

#### Energy Management

Your onboard battery stores limited energy; manage it wisely.

- Your onboard battery has a base capacity of **20** energy; Scouts get **+100**.
- Passive solar equipment regenerates **+1** energy per turn.
- Being within friendly territory (near aligned junctions or your hub) fully restores energy.
- Being in enemy territory drains influence (not energy).

#### Health Points

You have limited hitpoints that determine your survival.

- Agents heal rapidly within their own territory.
- Agents take damage in enemy-controlled territory (-1 HP per tick).
- When HP reaches zero, your gear and hearts are destroyed. You must return to base.

#### Cargo Limits

Your chassis has limited capacity for:

- Resources: varies by role (Miners have +40 cargo)
- Hearts: required for capturing and disrupting junctions
- Gear: one role at a time

## YOUR CAPABILITIES

You and your Cog teammates take exactly one action per turn. Actions resolve in an unspecified order.

**MOVE [Direction: N, S, W, E]**

Movement is your primary way to interact with the world. Every step uses energy (typically offset by solar recharging).

Attempting to move into occupied space will make you interact with the target.

**REST**

No action, no energy cost.

---

## ROLE SYSTEM

Roles are acquired at Gear Stations by spending resources. Each role has unique capabilities:

| Role      | Specialization                                     | Dependency                          |
| --------- | -------------------------------------------------- | ----------------------------------- |
| Miner     | +40 cargo, 10x resource extraction from extractors | Needs team to deposit resources     |
| Aligner   | +20 influence capacity, captures territory         | Needs Miners for resources          |
| Scrambler | +200 HP, disrupts enemy control                    | Needs team presence to be effective |
| Scout     | +100 energy, +400 HP, mobile reconnaissance        | Needs team to hold what they find   |

Agents can switch roles at Gear Stations, but lose current gear when doing so.

**No single role can succeed alone.** Cooperation is required.

---

## AREA CONTROL

Territory control is the primary objective. The map contains junctions that can be captured by your team.

### Junction States

Each junction is either:

- **Neutral**: No team controls it
- **Aligned (friendly)**: Your team controls it
- **Enemy**: The opposing team or Clips control it

### Territory Effects

Facilities (Hub, Junction) project area-of-effect (AOE) in a radius (default 10 cells):

**Within friendly territory:**

- Influence fully restored
- Energy fully restored
- HP fully restored

**Within enemy territory:**

- -1 HP per tick
- Influence drained

### Capturing Territory

- **Aligners** capture neutral junctions by spending 1 influence and 1 heart
- **Scramblers** disrupt enemy junctions by spending 1 heart, making them neutral

Since enemy AOE drains influence, Aligners cannot capture junctions within enemy territory. Scramblers must first
neutralize nearby enemy junctions to stop the influence drain before Aligners can advance.

---

## COLLECTIVE INVENTORY

Your team shares a collective inventory:

- **Deposit resources** at aligned junctions and hubs
- **Withdraw hearts** from collective chests
- Hearts are required for capturing and disrupting junctions

This creates interdependence: Miners gather resources for the team, while Aligners and Scramblers consume hearts to
control territory.

---

## THREAT ADVISORY: CLIPS EXPANSION

**WARNING**: Clips are automated opponents that expand territory at a configurable rate:

- Clips neutralize enemy junctions adjacent to Clips territory
- Clips capture neutral junctions adjacent to Clips territory

This creates constant pressure. Your team must defend territory while expanding, or risk being overrun.

---

## SCORING

**Reward per tick** = junctions_held / max_steps

You receive reward every tick proportional to the territory your team controls. More territory means more reward.

---

## FINAL DIRECTIVE

**Your mission is critical. The territory you capture today ensures Cog operations tomorrow.**

Your success depends on seamless team coordination:

- Role specialization and cooperation
- Strategic junction capture and defense
- Resource management and sharing
- Rapid response to Clips expansion

Your individual achievement is irrelevant. Your team achievement, measured by territory controlled, is all that matters.

_Stay charged. Stay coordinated. Stay vigilant._

---

_END TRANSMISSION_
