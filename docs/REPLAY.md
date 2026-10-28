# Replay format (`inbound-twin-replay/1`)

A replay is one day of the twin, exported as JSON under two policies, for any viewer to play back: the Blender video, the web viewer, or a plant's own tools. The viewer only interpolates. Every position, time and number comes from the twin.

Export one with `scripts/export_replay.py --regime R2 --seed 115 --day 10`. Find good days with `--find`.

All times are minutes from 00:00 of the replayed day. Negative values mean the previous evening.

| Key | Content |
|---|---|
| `meta` | regime, seed, day, the two policies in words, line (JPH, takt, production hours), plant (doors, crews, tugger loop and capacity, hot-run time), fleet, line-down $/min. `synthetic: true` |
| `network` | the plant at (0, 0); suppliers with x/y km, p_late and their parts; parts with supplier, usage/vehicle, units/container, kanban cards; `road_km`, the road distance matrix (node 0 = plant) |
| `traffic` | the true congestion multiplier of each supplier's corridor, every 15 min. The controller never sees it; it is there to colour roads |
| `plans` | the Phase 2 plans (`naive`, `routing`, `optimized`): truck, dispatch hour, supplier order |
| `runs.static`, `runs.twin` | one run record each (below), same seed: common random numbers |
| `story` | where to point the camera: the first stop under the static plan, its part, supplier and late truck, and the twin's first action and its lead time. Derived, not hand-picked |
| `headline` | study-wide results, read from `outputs/phase2` and `outputs/phase3`, with their source files |

**Run record**

| Key | Content |
|---|---|
| `trucks[].track[]` | `t0`, `t1`, `state`. The states are `travel` (`from`, `to` nodes), `load` / `late` (`at` a supplier), and `gate`, `yard`, `door_wait`, `unload`, `dock_out` (at the plant). Gaps between entries mean the truck is idle at the plant |
| `dock[]` | every truck at the dock, milk runs and other carriers: arrival, yard, door, crew, unloaded and leave times, containers, and a door number. Doors are one pooled resource in the twin; numbers are assigned first-free, only so a viewer can draw them |
| `receipts[]` | truck loads reaching the supermarket |
| `couriers[]` | premium freight: supplier; times of dispatch, arrival at the supplier, pickup, return to the plant and receipt; containers |
| `tugger[]` | tugger departures from the supermarket and drops at the line, with containers |
| `hot_runs[]` | forklift runs from the supermarket straight to the line: part, start, drop, containers |
| `line_stops[]` | start, end and the parts that were short |
| `cover_h` | every 5 min, each part's stock at the line and in the supermarket, in hours of production |
| `actions[]` | the twin's executed actions: time, kind, truck / supplier / part, premium $, and the decision label |
| `risk[]` | at every 15-min check, the least projected slack (minutes of use) and the part that has it |
| `day_kpis` | vehicles, line-down minutes and $, premium, disruption, logistics and total $ for the day |

**Adopting it.** A plant can produce the same file from its own data. It needs:
- a supplier list with coordinates;
- truck GPS states from a TMS;
- dock and receiving times from a WMS/YMS;
- line stops and part shortages from the MES.

Any viewer of this format then replays that plant's day. Running the twin itself on a plant's data needs `config/base.yaml` and the network generator replaced by that data. That adapter is not built yet.
