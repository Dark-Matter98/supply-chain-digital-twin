# Supply Chain Digital Twin

A discrete-event digital twin of an automotive-style inbound supply chain: milk-run trucks collect parts from suppliers. The parts then pass through a shared receiving dock, a supermarket and a tugger loop before reaching the assembly line.

The goal is to answer two questions with one model:
- How do you model the whole inbound system and reduce its total cost?
- When is dynamic re-routing of the trucks worth it?

The plant is synthetic, and every number is illustrative.

The full design is in [docs/SPEC.md](docs/SPEC.md).

## Status

| Phase | Content | Status |
|---|---|---|
| 1 | SimPy twin, (R,S) pickups + kanban, cost model, KPIs, tests | done |
| 2 | Static milk-run optimization (OR-Tools): frequency, routes, fleet; checked in the twin | done |
| 3 | GPS pings, Kalman filter, fast predictor, trigger x value-gate policies, experiment | done |
| 4 | Control tower (three.js), stress tests, video | done |

## Why model the plant, not just the trucks

A truck at the gate does not prevent a line stop. The material still has to get a door, wait for an unload crew, be unloaded and received, and catch the next tugger loop. The twin follows every container on that path. Dock capacity is shared with other carriers, so a late truck can delay others.

Two days of the default run (seed 1) show why this matters:

- **Day 9.** Supplier 6 wasn't ready for 2.8 hours. Truck 2 waited there with supplier 5's parts on board. Part 8 ran out at the line at 15:46, and the line stopped for 59 minutes. Sending the truck to the plant first would have helped.
- **Day 29.** Supplier 1 itself was 3.1 hours late, so its own part ran out at 10:00, and the line stopped for 105 minutes. Re-routing could not have helped that part.

Under the fixed plan nothing reacts on either day. Phase 3 is about telling these cases apart: when a re-plan pays, and when it can't.

## Run it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

```bash
.venv/Scripts/python -m pytest -q
```

```bash
.venv/Scripts/python scripts/run_phase1.py --days 30 --seed 1
```

```bash
.venv/Scripts/python scripts/run_phase2.py --days 30 --seeds 20
```

```bash
.venv/Scripts/python scripts/validate_phase3.py --gate 1
```

```bash
.venv/Scripts/python scripts/validate_phase3.py --gate 2
```

```bash
.venv/Scripts/python scripts/run_phase3.py
```

```bash
.venv/Scripts/python scripts/run_phase3.py --trigger-h 6
```

What each run writes:
- **Phase 1** writes `outputs/phase1/`:
  - `kpis.csv` and `costs.csv`: one row per day;
  - `events.csv`: the event log;
  - figures 1 to 3.
- **Phase 2** builds three plans: the naive one, an OR-Tools routing-only plan, and a plan that also optimizes frequency and fleet. Building them takes about 8.5 minutes. It then runs each plan in the twin on 20 held-out seeds (101 to 120), which takes about 3 minutes, and writes `outputs/phase2/` (`plans.csv`, `daily.csv`, `compare.csv`, figures 4 to 6).
- **Phase 3** reads the optimized plan from `outputs/phase2/plans.csv`, so run Phase 2 first.
  - The two `validate_phase3.py` gates check the estimator and the predictor on the tuning seeds before any experiment.
  - `run_phase3.py` tunes each policy's threshold on seeds 1 to 5, tests every policy on seeds 101 to 120, and sweeps the safety stock. It writes `outputs/phase3/` (`tuning.csv`, `test.csv`, `compare.csv`, `hypotheses.csv`, `ucurves.csv`, figures 7 to 11) and prints a hash of the results.
  - `--trigger-h 6` is the post-hoc run (`outputs/phase3_lookahead6/`).
  - `--figures-only` redraws the figures from the saved CSVs.
  - `--quick` is a small smoke run.

## Phase 2 result

Each plan was run on the same 20 held-out seeds for 30 days. Differences are vs the naive plan, with 95% bootstrap intervals:

| | Routing only (OR-Tools) | Routing + pickup frequency + fleet |
|---|---|---|
| Logistics cost/day | -$217 (-2%) | **-$1,102 (-11%)** |
| Line-down min/day | -1.7 (not significant) | **-5.3** |

- **Routing alone barely helps.** Drivers are paid by the shift, and the fleet cannot shrink, so shorter routes save km but no shifts or trucks.
- **Frequency is where the saving is.** Ten suppliers get fewer, fuller pickups. That cuts km by a third and removes one driver shift, and it outweighs the $327/day of extra inventory.
- **The twin caught a policy bug.** On the first evaluation the optimized plan looked like a disaster, with +28 line-down min/day. The cause was the rule that trims orders when a truck is full: under mixed pickup frequencies it always trimmed the least frequently picked supplier. After the fix, the optimizer's promise held. It predicted a $1,139/day saving, and the twin measured $1,102.

Details, deviations and what is not built are in [docs/SPEC.md](docs/SPEC.md) §16.

- `fig4_routes.png`: the 09:00 wave under each plan.
- `fig5_costs.png`: logistics cost by term, and line-down with intervals.
- `fig6_promise_vs_twin.png`: the optimizer's deterministic estimate vs what the twin measured.

## Phase 3 result

The question is when it is worth changing the plan during the day, and whether modeling uncertainty beats a simple stockout rule.

Each policy is a trigger (when to look) plus a gate (when to act). Ten policies run on top of the Phase 2 plan with 2 hours of safety stock, on the same 20 held-out seeds, in two worlds:
- R1: the Phase 1/2 world.
- R2: drifting traffic, incidents, and suppliers that differ in reliability.

Line-down is priced at $5,000 a minute.

| Total cost per day vs never re-planning | R1 | R2 |
|---|---|---|
| Simple stockout rule, act whenever it fires | -$13.1k | -$14.8k |
| Bayesian trigger + "act only if it pays" gate, as pre-registered (looks 2 h ahead) | -$3.2k (not significant) | -$7.2k (not significant) |
| Same, post hoc, looking 6 h ahead | -$11.5k | -$18.0k |
| Bayesian trigger, act whenever it fires, post hoc, 6 h | -$13.1k | **-$20.2k** |

- **The pre-registered test went to the simple rule.** The hypothesis that a Bayesian trigger wins in the drifting world was not supported.
- **A review after the run found a confound in the design.** The simple rule looked 6 hours ahead, the Bayesian trigger only 2. A courier can need up to about 3 hours, so the Bayesian trigger often fired too late.
- **With equal lookahead, the probabilistic trigger wins in the drifting world.** It beats the simple rule by $6.9k/day under the gate and $5.3k/day without it. In R1 it ties. This run was chosen after seeing the first results, so it is reported as post hoc, next to the pre-registered result.
- **The gain comes from treating arrival times as distributions, not from the Kalman updating.** An arm with one fixed, never-updated variance does as well. A corridor sees only a few truck legs a day, too few for the filter to track an incident (figure 7).
- **Material at the line is the lever, not truck routes.** Under the winning policies, trucks were re-routed at most 0.04 times a day. Couriers and hot runs from the supermarket to the line did the work.
- **"Act only if it pays" loses money here.** The gate cuts actions by 35 to 42%. It costs $1.7k to $4.0k a day more in every comparison, because actions are cheap next to a $5,000 minute of stoppage.
- **Bigger buffers make most of it unnecessary.** At the cheapest buffer (3 to 4 hours), dynamic re-planning saves only $40 to $210 a day. At 0.5 to 1 hour it saves $47k to $166k a day. It pays where a plant runs lean on purpose.
- **Open issue.** Looking 6 hours ahead, the periodic and event-triggered policies act 10 to 35 times a day. Under the gate, they also stop the line more often than never re-planning in R2. The predictor's scoring of candidate actions is not trustworthy when it is used that often. This is not diagnosed yet.

Details, all ten policies, the calibration of the predictor at 2 and 6 hours, and every deviation from the plan are in [docs/SPEC.md](docs/SPEC.md) §17.

Figures are in `outputs/phase3/`, and the post-hoc versions in `outputs/phase3_lookahead6/`:
- `fig7_filter_vs_truth.png`: the Kalman filter tracking a corridor through a traffic incident, against the hidden truth.
- `fig8_predictor_reliability.png`: predicted vs observed chance of a line stop, with the number of checks per bin.
- `fig9_arms.png`: total cost and executed actions per day for every policy.
- `fig10_ucurves.png`: total cost vs safety stock, with each policy's cheapest buffer.
- `fig11_decision_trace.png`: one drifting-regime day, the trigger's probability, the actions taken, and the line stops with and without re-planning.

## Stress tests

What happens when something breaks? Each stress test puts one disruption on one day (normal traffic) and runs the same 20 held-out days under the fixed plan and under the Shortage Early Warning. The figures are means for the disruption day.

| Disruption | Fixed plan: days the line stopped, min down | Early warning: days stopped, min down | Cost of the day |
|---|---|---|---|
| None (baseline) | 3 of 20, 3.7 min | 0 of 20, 0 min | $27.5k → $8.8k |
| Supplier 10 shut 08:00-16:00 | 20 of 20, 185 min | 20 of 20, 113 min | $933k → $574k (−39%) |
| Road to supplier 5 at a crawl (3× travel time), 07:00-13:00 | 20 of 20, 89 min | 14 of 20, 34 min | $453k → $179k (−61%) |
| Two of four dock doors out, 06:00-14:00 | 3 of 20, 6.7 min | 0 of 20, 0 min | $42.5k → $8.8k |

- A closed supplier cannot be fixed from the plant. Couriers wait for it too, so the early warning only buys time. The real answer is a second source.
- A slow road is where early action pays most: the warning sees the trucks running late and sends couriers or rushes stock before the line runs dry.
- Losing half the dock for a morning barely matters. The yard queue clears before any part runs out.

```bash
.venv/Scripts/python scripts/stress_test.py
```

It writes `outputs/stress/` (`days.csv`, `summary.csv`) and exports the median day of each scenario as a replay, which the control tower's scenario picker loads.

## Control tower and showcase video

`showcase/web/` is a three.js control tower for one simulated day, read from a replay file. It shows:
- the supplier network, with roads coloured by traffic;
- two plant floors side by side: the same day run as planned, and run with the twin;
- a live dashboard: line status, part 19's stock chart, line-down cost, alerts and actions, and the trucks.

The two runs share their random numbers, so they are identical until the twin's first action at 12:15. Every position and number comes from the replay.

To open the page, serve the project folder and open it in a browser:

```bash
.venv/Scripts/python -m http.server 8765
```

Then go to `http://localhost:8765/showcase/web/index.html`. It plays live, with play/pause, speed, a time scrubber and a Results button.

The video is recorded from the same page. `showcase/record.py` steps its scripted story (`showcase/web/story.js`) frame by frame in Chrome through Playwright, and encodes with ffmpeg:

```bash
.venv/Scripts/python showcase/record.py --aspect 16x9
```

Use `--aspect 4x5` for the feed version. Outputs go to `outputs/showcase/twin_control_tower_<aspect>.mp4`.

### Earlier Blender cut (kept as a fallback)

A 42-second 3D video of the same day. It has three stages:
1. **Export a replay.** `scripts/export_replay.py --find` lists candidate days; the export writes the day to JSON. The format is in [docs/REPLAY.md](docs/REPLAY.md).
2. **Render in Blender.** `showcase/blender_scene.py` builds the scene in code and renders frames headless.
3. **Add the overlays and encode.** `showcase/compose.py` draws the captions and HUD and encodes the MP4 with ffmpeg.

The edit is in `showcase/timeline.py`, and the storyboard in [docs/showcase/STORYBOARD.md](docs/showcase/STORYBOARD.md).

```bash
.venv/Scripts/python scripts/export_replay.py --regime R2 --seed 115 --day 10
```

```bash
"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b --factory-startup -P showcase/blender_scene.py -- --replay showcase/web/data/replay_R2_s115_d10.json --animate outputs/showcase/frames_16x9 --aspect 16x9
```

```bash
.venv/Scripts/python showcase/compose.py --frames outputs/showcase/frames_16x9 --aspect 16x9
```

Use `--aspect 4x5` for the feed version. Rendering takes about 0.35 s per frame on an RTX 4070 laptop GPU.

## Phase 1 figures

- `fig1_inventory.png`: hours of cover for three parts over one day, split by supermarket, line-side rack and inbound, with line-down shaded.
- `fig2_truck_gantt.png`: each milk-run truck's day, split into driving, loading, supplier delays, waiting at the plant, and unloading.
- `fig3_dock.png`: doors occupied, crews unloading, and the yard queue.

## Layout

```
config/base.yaml        every parameter (illustrative)
config/drifting.yaml    the R2 overlay: drifting traffic, incidents, uneven suppliers
src/inbound_twin/
  config.py             typed config + validation
  rng.py                keyed random draws (common random numbers across policies)
  network.py            synthetic suppliers, parts, distances, kanban sizing
  scenario.py           traffic, supplier delays, handling times, other carriers
  plan.py               milk-run plans, route timing and planned lead, naive baseline, (R,S) quantities
  state.py              TwinState: all simulation state, snapshot()
  sim.py                the SimPy twin
  costs.py, kpi.py      cost model (driver shifts, overtime, trucks), daily KPIs, Little's law check
  optimize.py           OR-Tools routing per wave, frequency search, fleet size, planning cost
  evaluate.py           runs plans and policies in the twin, paired bootstrap comparisons
  observe.py            what the controller may see (no truth), GPS pings
  estimate.py           Kalman filter over corridors, supplier readiness, bayes/det/fixed views
  predict.py            fast vectorized predictor of the next 6 h, same scenarios for every candidate
  policy.py             triggers, gates, candidate actions, frozen zone
  diagnostics.py        loggers for the stop gates and figures (reads truth, scoring only)
  plots.py              figures
  stress.py             stress-test scenarios (supplier outage, road closure, dock doors out)
scripts/run_phase1.py, run_phase2.py, validate_phase3.py, run_phase3.py, export_replay.py, stress_test.py
showcase/
  web/                  three.js control tower (index.html, app.js) and the video's script (story.js)
  record.py             records the control tower's story to MP4 (Playwright + ffmpeg)
  timeline.py           the Blender cut's edit: beats, simulated time, camera, captions, labels
  blender_scene.py      builds the 3D scene from a replay and renders frames (Blender's Python)
  compose.py            captions, HUD and labels over the frames; MP4 via ffmpeg
tests/                  68 tests: hand-checked timeline, conservation, snapshot, Little's law, extremes,
                        calibration, shift pay, lead estimator vs twin, trimming rule, brute-force
                        optimality of the router, no truth in observations, filter coverage,
                        hand-checked predictor and value gate, frozen zone, courier and hot-run
                        material balance, stress tests
```
