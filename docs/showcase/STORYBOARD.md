# Showcase video: storyboard

The 42-second Blender cut. It shows one real day replayed from the twin, then the study-wide numbers.

- Source: `showcase/web/data/replay_R2_s115_d10.json` (`scripts/export_replay.py --regime R2 --seed 115 --day 10`).
- Accuracy: every movement and time below comes from that file, and every study-wide number from `headline` in the same file.
- Hero day: chosen by `scripts/export_replay.py --find` as the longest stop that the twin prevented with a visible cause. It is one illustrative day, not a typical one. The study-wide numbers in beat 6 are the typical result.
- Formats: 16:9 and 4:5 (phone feeds). Captions are burned in, because most people watch muted.

## The day (drifting-traffic world, seed 115, day 10)

| Time | Static plan (never re-plan) | With the twin (det + always) |
|---|---|---|
| 09:00 | Truck 4 leaves for suppliers 5 and 10 | same |
| 09:00-13:48 | Corridor traffic builds to 1.3-1.6x. Legs take 2.4x and 2.6x nominal (149 min vs 61, 115 vs 44). Truck 4 reaches supplier 10 about 2.6 h late | same: the twin cannot change traffic |
| 12:15 | | The twin's projection: part 19 (supplier 10) will run **22 min short** within 6 h, about 3 h before it would happen |
| 12:15-13:00 | | Priority door for truck 4; **hot run** of part 19 straight to the line (12:44); **courier** to supplier 10 for part 19 (sent 12:45, back 13:44, $383); a second courier, for part 18 |
| 14:19 | Truck 4 at the gate; unloaded 15:20 | same |
| **15:10** | **Part 19 runs out. The line stops** | Line running: part 19 had 1.8 h of cover at the line at 15:00 |
| 15:43 | Truck 4's load is in the supermarket, but the line still waits for the tugger | |
| 16:15 | Line restarts after **64 min down ($321,000)** | |
| Day | 747 of 800 vehicles; $329.7k total | **800 of 800**; $9.8k total, of which actions cost $1.1k |

## Beats (about 40 s at 30 fps)

The exact edit (seconds, simulated time, camera keys, captions, labels, which HUD pieces show) lives in `showcase/timeline.py`, which both renderers read. This table summarizes it.

| Video time | Beat | Simulated time | What is on screen |
|---|---|---|---|
| 0-4 s | Hook | 08:50-09:06, static plan | Aerial push-in over the network as the 09:00 wave leaves. "What if your plant could see a line stop 3 hours before it happens?" |
| 4-11 s | **System + process** | 10:25-11:35 | Fly into the plant. Labels: supplier, milk-run truck, dock doors, supermarket, tugger train, assembly line. "A digital twin follows every container" |
| 11-15 s | **Problem: supply chain** | 09:00-13:50 | Chase camera on truck 4 through amber and red corridors. "Traffic slows one truck" |
| 15-19 s | **Problem: the line** | 13:50-15:14 | Push in on part 19's line rack. It turns amber, then the red plate shows; the andon turns red at 15:10. HUD: stock gauge, line status, line-down counter |
| 19-22 s | **System insight** | 15:14-16:20 | Truck 4 at the dock, the load in the supermarket at 15:43, the line restarts 16:15. "A truck at the gate is not a part at the line". Counter ends at 64.2 min, $321,000 |
| 22-23 s | Rewind | 16:20 back to 12:00 | "Same day. Same traffic. Now with the twin." |
| 23-25.5 s | **The twin sees** | 12:00-12:32, twin | The forecast appears in the action log at 12:15; label "Part 19: projected 22 min short" |
| 25.5-28.5 s | **The twin acts** | 12:32-13:45 | Wide view west: couriers to supplier 10 and back; truck 4 still in traffic. Action log fills |
| 28.5-31 s | **It holds** | 13:45-15:20 | The same push-in on part 19's rack; 15:10 passes, the line stays green. Counter at 0 |
| 31-37 s | **Math programming + results** | plan overlays | Naive routes (60 pickups/day), then optimized (37); cards: -11% logistics cost; about half the total cost with re-planning (-53% / -49%, 20 test runs each) |
| 37-42 s | End card | | Title, "Supply chain · plant process · optimization · system decisions", stack, name, "Synthetic plant and data. Every movement replayed from the simulation." |

## Visual rules

- Stylized low-poly look, no photoreal. Everything is built in code; no downloaded assets.
- Colors carry meaning:
  - neutral grey for normal flow;
  - amber for risk;
  - red only for a stopped line and the cost of the stop;
  - one accent colour for the twin's actions (courier, hot run, HUD).
- One idea per beat. Hold every caption at least 2 s. Numbers are always rounded the same way as in SPEC section 17.
