# Supply Chain Digital Twin: design spec

Status: Phases 1 to 3 built and run (results in sections 15 to 17). Phase 4 built: the three.js control tower, the stress tests (section 18) and the video.
All numbers are illustrative. The plant is synthetic and uses no data from any real company.

## 1. Purpose

The project answers two questions that come up in industrial supply chain interviews:

1. How would you model the whole inbound system and reduce its total cost?
2. How would you run dynamic routing for the delivery trucks?

It does this with one model: a discrete-event twin of the path from supplier to assembly line, a cost model over that path, a static routing optimizer, and a dynamic layer that decides when re-routing is worth it.

What it does not claim: algorithmic novelty. Dynamic vehicle routing and re-optimization have a large literature (Pillac et al., 2013, *A review of dynamic vehicle routing problems*, EJOR, is the starting point). The contribution is the formulation, the integration with plant-side material flow, and a careful evaluation.

## 2. System and flow

A part prevents a line stop only when it is at the line. Truck arrival is not enough. The twin follows every container:

```
supplier --truck--> gate --> yard --> door [wait for crew] [unload] [dock out]
   --> staging --receiving--> supermarket --tugger--> line-side rack --> consumed
```

- Milk-run trucks share the dock with other carriers' trucks (background traffic). That traffic uses doors and crews, so one late truck can delay another.
- The tugger runs a fixed loop (default 30 min). It leaves the supermarket on the loop, carries every pending kanban it can fill, and drops them at the line half a loop later. A container received one minute after the tugger left waits a full loop.

## 3. Entities, resources, state

| Item | Default | Notes |
|---|---|---|
| Line | 1 line, 50 JPH, 06:00 to 22:00 | 800 vehicles/day if never starved; takt 1.2 min |
| Parts | 30 (2 per supplier) | usage 1, 2 or 4 units/vehicle; units/container set so each part uses 0.5 to 1.1 containers/h |
| Suppliers | 15, 10 to 65 km straight-line, road = 1.3 x straight-line | 60 km/h base speed |
| Trucks | 5, 26 containers each | one container = one pallet position |
| Dispatch waves | 04:00, 09:00, 14:00, 19:00 | every truck, every wave (naive plan) |
| Doors / crews | 4 / 2 | a truck holds its door while it waits for a crew |
| Background trucks | Poisson 1.5/h, 04:00 to 24:00, 10 to 26 containers | use doors and crews only |
| Tugger | 16 containers, 30-min loop | runs around the clock |
| Kanban cards | 2 or 3 per part | see section 5 |

All state lives in `TwinState` (`src/inbound_twin/state.py`): inventory per part in each location (`on_order`, `in_transit`, `in_plant`, `staging`, `supermarket`, `tugger`, `line_full`, `line_open`), the kanban queue, trucks, yard, door and crew counts, and line status. Generator code keeps only timing. `snapshot()` is therefore a complete picture of the plant, which the Phase 3 predictor reads.

## 4. Uncertainty and random numbers

| Quantity | Distribution | Key |
|---|---|---|
| Leg travel time | nominal x rush factor (1.35 at 07-09 and 16-18) x lognormal(mean 1, sigma per corridor) | (day, from, to, truck, 15-min departure bucket) |
| Supplier not ready | Bernoulli(0.05), then exponential(mean 40 min) | (day, supplier, hour of arrival) |
| Unload time | lognormal, mean 5 + 1.2 x containers, cv 0.2 | (day, truck, trip number) |
| Receiving time | lognormal, mean 20 min, cv 0.3 | (day, truck, trip number) |
| Background trucks | Poisson count, uniform times, uniform sizes | (day) |

Every draw comes from its own generator, seeded by `(seed, kind, key...)` (`rng.py`). A draw depends on what physical event it describes, not on the order the simulation asks for it. Two runs with different plans therefore see the same traffic and the same supplier delays for the same event. That gives common random numbers across policies, so paired comparisons in Phase 3 have low variance. Keys are integers; Python's string `hash()` is salted per process and is never used.

Simplification: a leg's travel time is fixed when it departs (rush factor at the departure time).

## 5. Inventory policy

**Pickup quantities: periodic review, order-up-to (R,S).** At dispatch, for each part on the route:

```
S_p(t) = rate_p x [ production_hours(t, next_dispatch + lead_of_that_route + lead_margin_h) + ss_h ]
order_p = ceil( (S_p - inventory_position_p) / units_per_container_p )      (>= 0)
inventory_position = on hand everywhere (truck, dock, staging, supermarket, tugger, line) + on order
```

- `production_hours` counts only 06:00 to 22:00, so nights and uneven waves are handled exactly. The waves are 3 to 5 production hours apart, so an average review period would under-cover the long gaps. Phase 1 started with that average, and the line stopped near 18:00 almost every day; the exact form removed that pattern.
- **Lead time comes from the plan.** Every route carries a planned `lead_min`, the minutes from dispatch until its load is in the supermarket. `plan.route_timing()` computes it from: nominal travel × the rush factor at each leg's departure, the expected not-ready delay per stop, loading, gate, a planned dock wait (13 min), the mean unload, and receiving. Plans with different route lengths therefore order against their own lead times. Phase 1 used one global 4.0 h; Phase 2 replaced it because a constant would bias the comparison of plans.
- **Lead and volume are solved together.** Loading and unloading make lead depend on volume. Volume depends on lead, because an order covers from its own load's arrival to the next load's arrival. `build_plan` iterates the two to a fixed point.
- **Estimator check.** On the naive plan the planned lead is within 7% of the twin's mean lead on every route (`test_planned_lead_matches_twin_mean`). The check compares to the mean, not the median, because the lead distribution is right-skewed. The largest gap (+6.9%) is a route whose planned return leg departs just after the 16:00 rush-hour step, while only a third of real departures do.
- `ss_h` = safety stock in production hours. This is the buffer knob for the Phase 3 sweep.
- **Trimming.** If a route needs more than 26 containers, containers are removed one at a time from the part that would keep the most safety hours at the end of its cycle. Safety hours = (position after the order − consumption until the next load lands) / rate. This evens out the remaining safety cover.
  - Phase 1 trimmed the part with the most raw hours of cover after the order.
  - With every supplier on every wave, the two rules agree. With mixed frequencies, the old rule always trims the least frequently picked parts first, because their orders must last longest.
  - In Phase 2 that sent 338 trimmed containers to one twice-a-day supplier over 3 seeds × 30 days and caused 33 of P2's 37 line-down min/day (`test_trimming`).

**Line-side kanban.** Cards per part:

```
K_p = max(2, ceil(d_p x 1.5 x loop x (1 + alpha)) + 1)
```

Here d_p is containers per minute, alpha = 0.5, and the +1 is the open container. The worst replenishment wait is 1.5 loops: up to one loop until the next departure, plus half a loop to the drop. Cards are conserved: `line_full + [open] + pending + on tugger = K_p` at all times. A test checks this.

## 6. Cost model

Daily cost (`costs.py`), all rates in `config/base.yaml`:

| Term | Formula | Default rate |
|---|---|---|
| Transport | km driven x rate | $1.20/km |
| Driver | per shift: 8 paid hours, plus 1.5 x the span beyond 8 h (span = first dispatch to last door release). Waves 04/09 form the early shift, 14/19 the late shift. | $35/h |
| Trucks | trucks in the plan's fleet x lease rate | $250/truck-day |
| Handling | containers unloaded x rate | $4/container |
| Holding, capital | sum over parts and locations of unit-hours x unit value x carrying rate / 8760 | 20%/yr, in-transit included (plant-arranged pickup) |
| Holding, space | (peak supermarket + staging positions after warm-up + line-side card positions) x rate | $1.50/position-day |
| Line down | minutes starved in the production window x rate | $5,000/min |
| Premium freight (Phase 3) | fixed + per km | $250 + $3/km |
| Disruption (Phase 3) | per executed change + per changed stop | $50 + $20 |

Space is a capacity cost: you pay for the racks you must provide, so it uses the peak, not the average.

Drivers are paid by the shift, idle time between waves included. A plan saves on drivers only by needing fewer shifts or less overtime, not by shaving minutes off a route that ends well inside the shift. (Phase 1 priced driver busy hours; Phase 2 replaced that.)

## 7. KPIs

Per day, excluding the warm-up day (`kpi.py`):
- Output: vehicles built, line-down minutes, line stops.
- Fleet: trips, km, driver hours, containers picked and unloaded, truck fill.
- Dispatch: dispatch lateness.
- Utilization: trucks, doors, crews (busy time over the 04:00 to 24:00 operating window); tugger (mean load / capacity on production-hour loops).
- Material flow: yard wait, gate-to-unload time, gate-to-supermarket time (container-weighted).
- Cover: mean supermarket cover, minimum part cover.

## 8. Decision layer

### 8.1 Static plan optimization (Phase 2, built: `optimize.py`)

**Decisions** for the daily template (the same plan runs every day):
- **Pickup frequency per supplier.** The options are which waves visit it: every wave; every other wave ({04, 14} or {09, 19}); or one wave. One wave is allowed only if a whole day's volume fits one truck.
- **Routes per wave.** Which truck visits which suppliers, and in what order.
- **Fleet size.**

**Routing model per wave.** A capacitated vehicle routing problem with a route-duration limit, solved with OR-Tools.
- **Demand.** The expected containers of each visit. An (R,S) order covers from this load's arrival to the next load's arrival, so this is the production hours in that window × the supplier's container rate. Demand is kept in tenths of a container.
- **Capacity.** 26 × `max_fill` per truck.
- **Duration.** Driving (× the wave's average rush factor) plus service must fit before the truck's next wave. The cutoff is the next dispatch, or 24:00 for the last wave, minus `slack_min` and minus the fixed part of the plant visit (gate, planned dock wait, unload setup, dock out). Service at a supplier = loading + that supplier's share of unloading at the plant + the expected not-ready delay. Unloading is linear in containers, so attributing it to pickups is exact.
- **Cost.** km × $/km + driving minutes × driver $/h, as an overtime proxy. Each truck used adds the fixed plant-visit time × driver $/h.
- **Search.** Every visit is optional at a huge penalty ($10M), so the model always has a solution. If the best solution found drops a visit, the wave counts as infeasible. That is a heuristic verdict, not a proof.
  - **Construction.** Savings.
  - **During the frequency search.** Greedy descent to a local optimum, about 0.01 s and within 1–5% of the final result on the default waves. If it drops a visit, guided local search gets a chance before "infeasible".
  - **Final solves.** Guided local search stopped after 2,000 solutions, 2–3 s per wave.
  - **Determinism.** All of it is deterministic.
- **Time limits are a safety net, not a stopping rule.** If a solve reaches one, the code raises an error rather than keep a result that would depend on CPU speed.
- **How it got here.** Each step was found by testing, not chosen up front.
  1. A short time limit used to declare infeasibility gave two different P2 plans in two runs under different machine load.
  2. Savings construction alone, as the feasibility test, stalled for 3–25 s on tight waves and wrongly called 5 of 8 sampled waves infeasible: guided local search found full solutions for them.
  3. Cheapest-arc construction failed on duration-tight waves where savings did not.
  4. Making visits optional removed the stalls.

**Search over frequencies and fleet (a heuristic decomposition, not a joint optimum).**
1. **Frequencies.** Greedy local search from "every supplier on every wave", using fast solves. Each move changes one supplier's option and re-solves only the waves whose demand changed. Infeasible assignments are ranked by total demand over capacity, so the search can climb toward feasibility.
2. **Fleet.** Fleet sizes are tried from 5 down while every wave stays feasible.
3. **Final solve.** Each wave is re-solved with full search effort.

Trucks are assigned within a driver shift so that a truck that ran the shift's earlier wave takes the shortest later route. That means no extra paid shift and the least overtime.

**Planning cost** (`planning_cost`): the optimizer's deterministic promise, with no line-down.
- Transport, driver shifts and overtime from planned route timings, truck lease, and handling.
- Holding capital from estimated safety, cycle, pipeline and line-side stock.
- Space from the sum of each part's own peak, which is an upper bound.

The twin then checks this promise under uncertainty (section 16).

**Assumptions.** Suppliers have no opening-hour windows. Returnable empties use no truck capacity. Wave dispatch times are fixed inputs, not decisions (see section 16 for why that matters).

**Baseline.** The Phase 1 naive plan: sweep sectors, one truck per sector, nearest-neighbour order, every wave.

### 8.2 Dynamic re-planning (Phase 3, built: `policy.py`)

Each policy has two parts: when to **look** (trigger) and when to **act** (gate).

| Trigger | Fires when |
|---|---|
| never | (baseline: the static P2 plan runs untouched) |
| periodic | every 30 min |
| event | leg done, supplier not ready, arrival at the plant, receipt, line stop |
| det | every 15 min, point-estimate projection: some part's least slack over the next 6 h < m minutes |
| fixed | every 15 min, P(line stops within 2 h) > tau, with one pooled, never-updated variance |
| bayes | every 15 min, P(line stops within 2 h) > tau, with the filter's variance and ping-sharpened legs |

- **Assessment.** The predictor ranks the at-risk parts (up to 3). Candidates are built for them, and the current plan and every candidate are scored in one predictor call on the same scenarios.
- **Gate.**
  - `value` executes the best candidate only if its expected cost, with disruption and premium included, is below the current plan's.
  - `always` executes the best candidate whenever parts are at risk.
- **Actions (v1).**
  - Road side, for the truck carrying the part or about to pick it up:
    - leave a not-ready supplier now, without its load (interrupts the wait);
    - go to the plant after the current stop;
    - move the part's supplier to the front of the remaining stops (then the rest, or straight to the plant);
    - priority door at the next plant visit.

    Dropped stops release their orders to the (R,S) rule for the supplier's next pickup.
  - Premium freight: a courier brings 2 production hours of the part from its supplier ($250 + $3/km round trip) and it is received express (10 min).
  - Plant side: a hot run sends the part's next containers from the supermarket straight to the line in 10 min, skipping the tugger loop.
- Not built: moving a stop to another truck, a spare truck, diverting a truck mid-leg, line-speed changes.
- **Frozen zone.** A truck at the plant, or under 30 min (posterior mean) from it, cannot be re-routed; plant-side actions still apply.
- **No flip-flopping.** At least 60 min between executed changes to the same truck.
- Activity tracked: assessments, executed actions by kind, and missed stops (a line stop with no trigger in the 60 min before it).

## 9. Estimation (Phase 3, built: `observe.py`, `estimate.py`)

- **Observation layer.** The controller only sees what a TMS/WMS would:
  - inventory by location;
  - each truck's status, current leg (from, to, start), load, remaining stops and orders;
  - GPS pings;
  - "supplier not ready since t";
  - batches in receiving, couriers in flight, recent dock waits, trips started.

  True leg end times and the corridor congestion path never leave the twin (`test_observation_has_no_truth_fields`).
- **GPS pings.** Every 10 min while a truck drives, it reports the fraction of the leg done, with noise sd 0.03 (keyed draws).
- **Kalman filter.** A joint linear-Gaussian filter over the 15 corridor log-multiplier offsets.
  - Model: AR(1) in 15-min steps. An arc observes the mean of its supplier endpoints.
  - Incidents are not in the model; their average effect is folded into the stationary mean and variance.
  - Parameters come from config, as if fitted on history.
  - Committed updates come from completed legs only. Out-of-sequence legs get noise inflated by the drift since departure.
  - The latest ping of each leg in progress updates a temporary copy, used to predict other legs, so pings are never counted twice.
- **Current leg.** The committed prior is combined with the leg's own latest ping (delta method), truncated at "has not arrived yet".
- **Supplier readiness.** Beta-Bernoulli per supplier from a common prior (20 pseudo-visits at the configured p_late). A delay already running is memoryless exponential.
- **Views for the ablation.**
  - `bayes`: mean + variance.
  - `det`: the same means, no spread.
  - `fixed`: the same means, one pooled variance, no ping sharpening of the spread.
- **Stop gate 1 (passed).** On tuning seeds 1-5 × 10 days, 80% intervals for the remaining leg time covered 80.5% (R2) and 80.1% (R1). The mean log score beat the prior (no filter, no pings): −2.50 vs −3.40 (R2) and −2.27 vs −2.89 (R1). In R1 the whole gain comes from pings; the filter is inert there because there is no drift to learn.

## 10. Fast predictor (Phase 3, built: `predict.py`)

- **Why not the full twin.** Re-running the SimPy twin in every assessment is too slow, so a reduced model is used and checked against the twin.
- **What it does.** It reads an Observation and simulates N = 200 scenarios of the next 6 h in vectorized numpy:
  - each truck finishes its current leg (posterior), waits at suppliers, loads, drives the remaining legs (filter predictive), waits at the dock (recently observed waits), unloads and receives;
  - then its scheduled trips not yet started, with expected volumes. A trip whose time has passed while the truck was busy leaves when the truck is free; the Observation's trips-started count says which trips those are;
  - plus batches in receiving and couriers in flight.
- **Line-side model.** Stock reaches the line one tugger wait after the supermarket (10 min with a hot run). A part is short whenever cumulative consumption exceeds what has reached the line. Line-down = production minutes in which any part is short.
- **Stated approximation.** A stopped line also stops consuming; the predictor ignores that.
- **Cost.** 4.7 ms per call (0.7 ms in point-estimate mode). Draws are keyed by (seed, day, call, event), so the current plan and every candidate see the same scenarios.
- **Stop gate 2 (passed, scored at the line level).** P(line stops within 2 h), logged every 15 min under "never" on tuning seeds 1-5 × 10 days:

  | Regime | Calibration slope | Brier | Climatology |
  |---|---|---|---|
  | R1 | 0.85 | 0.0011 | 0.0135 |
  | R2 | 0.92 | 0.0008 | 0.0152 |

  When it said ≥ 0.9, the line stopped within 2 h every time (57/57 and 60/60).

  Per part the outcome is censored: the twin records only the parts short when the line stops, and a part about to run out behind another never "starves". The per-part slope (0.60) is reported for reference, not gated.
- **Caveat on the slope criterion.** A logistic slope is undefined when predictions separate the outcomes perfectly. A 4-day, 1-seed smoke run did exactly that (Brier 0) and showed slopes of 3.7-5.7. The gate is meaningful only on enough data.
- **At 6 h** (checked after the experiment, for the post-hoc run in section 17.3), the gate passes in R2 and fails in R1. Figure 8 shows the reliability curves at both horizons.

## 11. Experiment design (Phase 3, built: `scripts/run_phase3.py`)

- **Base plan.** P2 (Phase 2's optimized plan), loaded from `outputs/phase2/plans.csv`. Dynamic re-planning is tested on top of the best static plan.
- **Regimes.**
  - R1 = the Phase 1/2 world.
  - R2 = `config/drifting.yaml`:
    - corridor AR(1) congestion (σ 0.2, φ 0.9 per 15 min);
    - incidents (0.1 per corridor-day, ×1.4-2.2, about 90 min);
    - leg-noise σ that differs by corridor (0.06-0.34);
    - p_late that differs by supplier (0.02-0.15).
- **Tuning (seeds 1-5, 15-day runs).** The best threshold for each trigger and gate, chosen by mean total cost at $5k/min:
  - det: m ∈ {−30, −15, 0, 15, 30, 60} min;
  - bayes: tau ∈ {0.05, 0.1, 0.2, 0.4};
  - fixed: the same tau grid, gate = value only.
- **Test (seeds 101-120, 15-day runs, day 0 excluded).** 10 arms per regime:
  - never;
  - periodic and event, each with both gates;
  - det and bayes at their tuned thresholds, each with both gates;
  - fixed + value.

  Paired with "never" through common random numbers, with bootstrap 95% CIs. Total cost is reported at $1k, $5k and $20k/min.
- **U-curves (seeds 101-110).** Safety stock ∈ {0.5, 1, 2, 3, 4} h for never, det + value and bayes + value, using thresholds tuned at 2 h.
- Hypotheses, stated before any results:
  - H1: in R1, bayes ≈ det (CI includes 0).
  - H2: in R2, bayes < det in total cost.
  - H3: the value gate cuts executed actions at equal or lower cost.
  - H4: dynamic policies move the cost-minimizing buffer left.
- **A fair deterministic baseline.** It uses the same pings and the same filter means as the Bayesian arm; only the spread differs. The comparison isolates the value of the uncertainty, not the value of the data.

## 12. Verification and validation (Phase 1)

| Test | What it shows |
|---|---|
| `test_rng` | Same key gives the same draw, also in a fresh process. Different keys give independent draws. |
| `test_network` | Generator is deterministic. Distances are symmetric and satisfy the triangle inequality. The naive plan visits every supplier once per wave. |
| `test_conservation_invariants` | Exact material balance per part. Consumption = usage x vehicles. Doors, crews, tugger and trucks stay within capacity, rebuilt from the records. Time stamps are ordered. The line never runs outside its window. A mid-run `snapshot()` balances and is independent of the live state. The same seed is deterministic. |
| `Twin.assert_invariants` | Checked every 5 simulated minutes in tests: non-negative stock, material balance, kanban cards conserved, resource counts. |
| `test_littles_law` | L from the state integral equals lambda x W from the batch records, within 2% (they match to 3 decimals). |
| `test_handcheck` | A 1-supplier, 1-part, 1-truck, no-noise case; every time stamp, the order quantity and the costs are worked out by hand in the test docstring. |
| `test_extremes` | No noise + big buffer gives zero line-down. No buffer + heavy noise gives line-down. More doors gives less yard wait. More safety stock gives more holding cost. |
| `test_calibration_costs` | Default utilizations fall in their bands. Cost formulas are checked by hand. Daily total = sum of terms. Bad configs are rejected. |
| `test_shift_cost` | Shift pay and overtime by hand. Shifts group the right waves. Planned lead is within 10% of the twin's mean lead on every naive route. |
| `test_trimming` | When a truck is too small for two parts with different pickup frequencies, both give up containers and their leftover safety cover ends within one container of each other. The old rule fails this test (checked by swapping it back in). |
| `test_world_model` | Drift is off by default. The corridor AR(1) has the stated std (0.2), lag-1 correlation (0.9) and a mean-one multiplier. The incident rate (0.1/day) and minimum jump size match. The drifting overlay keeps the same plant and suppliers. |
| `test_observe_estimate` | The Observation has no truth fields. Pings stay within the leg and the clock, and are deterministic. The filter's 80% predictive intervals on legs drawn from the world's own process cover 74-86%. The filter is inert without drift. The mean remaining leg time matches Monte Carlo. |
| `test_predict_policy` | A 1-part predictor case by hand (120 min short, P(stop in 2 h) = 0, margin -120 min) in both views. A trip that is overdue but not started is still predicted to come. Every candidate sees the same scenarios. The value gate is checked by hand in three cases. The frozen zone: at the plant, recently changed, under 30 min out. Leaving a not-ready supplier keeps the books, from the event path and from the interrupt path, with invariants checked every 5 min. Courier and hot run conserve material. The same seed gives the same decisions. |
| `scripts/validate_phase3.py` | Stop gates 1 and 2 (sections 9 and 10). |
| `test_optimize` | Frequency options. Visit volumes by hand. OR-Tools equals a brute-force optimum (every partition, every order) of the same integer model on a 6-supplier case. Default waves are feasible (each supplier once, capacity, duration) and the solver is deterministic. Planned km equals twin km. An optimized plan respects its frequencies and runs in the twin with invariants checked. |

Verification asks whether the code does what this spec says; the tests above cover it. Validation asks whether the model is fit for the question. With synthetic data, validation is limited to face validity (figures 1 to 3 make operational sense), extreme-condition behaviour, and calibration to stated utilization bands.

## 13. Assumptions and non-goals

- Synthetic network and illustrative rates. No real company data.
- Single assembly line. No line-speed changes or model mix. Line-speed (JPH) changes were planned as a Phase 3 event and were not built.
- One container = one pallet position, for every part.
- Suppliers always have stock; their only failure is not being ready on time.
- Drivers are paid by the shift (8 h minimum, overtime beyond). There are no hours-of-service limits beyond the route-duration cap.
- Background trucks carry parts outside the model; they only use dock capacity.
- Travel time is fixed at departure.
- Lost production is not made up with overtime. The cost is the line-down rate.
- **Phase 3.**
  - The controller sees only what a TMS/WMS would (section 9). The filter's parameters are given, as if fitted on history, not estimated in the run.
  - Couriers are always available at the stated price, with no limit on how many are out. A courier drives in the same traffic as the trucks and also waits if its supplier is not ready.
  - An assessment costs nothing. Only an executed action carries disruption cost.
  - The predictor scores only the next 6 h and keeps consuming while the line is stopped (section 10). Anything an action pushes past 6 h is not in its score.
  - Thresholds are tuned at 2 h of safety stock and reused at every buffer level in the U-curves.

## 14. Phases

| Phase | Builds | What you can say afterwards |
|---|---|---|
| 1 (done) | SimPy twin of supplier to line, (R,S) + kanban, cost model, KPIs, tests | The whole inbound path as queues; where material waits; why truck arrival is not material availability; landed logistics cost per day and what drives it |
| 2 (done) | OR-Tools routing per wave, pickup frequency vs holding cost, fleet size, checked in the twin | Routing formulation, solver, an 11% logistics saving that the twin confirmed, where it comes from, and why the optimizer's promise must be checked under uncertainty |
| 3 (done) | GPS pings, Kalman filter, fast predictor + calibration check, trigger x gate policies, experiment | Dynamic re-planning with numbers: the best policy halves total cost at a 2 h buffer, mostly through plant-side actions and couriers, not truck re-routes; a simple stockout rule won the pre-registered test; after fixing a lookahead confound (post hoc), a probabilistic trigger won in the drifting regime, and the gain came from modeling uncertainty, not from the Kalman updating; and larger buffers make most of this unnecessary |
| 4 | Dashboard (map, KPIs, what-if sliders), README figures | A demo you can share |

## 15. Phase 1 results (naive plan, seed 1, 30 days, day 0 excluded)

These are the numbers after the Phase 2 model changes: plan-derived lead times, paid driver shifts, truck lease, and the new trimming rule (section 16). The original Phase 1 numbers are in the deviations list below.

| KPI | Mean per day |
|---|---|
| Vehicles built | 793.1 of 800 |
| Line down | 8.4 min (0.24 stops) |
| Truck / door / crew / tugger utilization | 0.77 / 0.41 / 0.56 / 0.72 |
| Truck fill | 0.73 |
| Gate to supermarket | 70 min |
| Driver shifts / overtime | 10 shifts / 8.1 h |
| Total cost | $51.8k, of which line-down $41.9k, driver $3.2k, transport $2.8k, handling $1.5k, trucks $1.25k, holding $1.2k |

**Day 9, the case for re-routing.** Supplier 6 was not ready for 2.8 h. Truck 2 waited there with supplier 5's parts on board. Part 8 ran out at 15:46, and the line stopped for 59 min until the load was received and the next tugger loop dropped it. A re-plan would have sent the truck to the plant first.

**Day 29, the case against.** Supplier 1 itself was 3.1 h late. Its part ran out at 10:00, and the line stopped for 105 min. No re-route helps the late supplier's own part. Only premium freight, a plant-side action, or accepting the stop is left. The Phase 3 value gate has to tell these two days apart.

**Open question for Phase 3: cost calibration.** At $5,000/min, line-down is about 80% of daily cost and holding is about 2%. In the original Phase 1 run, raising safety stock from 2 to 3 hours added about $170/day of holding and cut line-down from $61.0k to $13.7k/day (seed 1). At this rate the U-curve puts the optimal buffer far to the right. Phase 3 has to show results across a range of line-down rates and state that the optimum depends on that rate.

**Original Phase 1 numbers** (global lead 4.0 h, drivers paid busy hours, no truck lease): 790.1 vehicles/day, 12.2 line-down min/day, $69.2k/day.

**Deviations from the original plan, and why.**
1. **Door band.** The plan's door band (50 to 60%) could not coexist with crews near 60% on 4 doors and 2 crews, since a door is held mostly while one of the two crews works. The band is now 35 to 50% for doors and 50 to 65% for crews.
2. **Background trucks added.** With only 20 milk-run arrivals a day on 4 doors, the dock would never queue, and the material-availability refinement would have nothing to show.
3. **(R,S) review period.** An average review period was replaced with exact production hours to the next pickup (see section 5). `lead_h` is now defined in clock hours.
4. **scipy dropped from dependencies.** Nothing in Phase 1 uses it.

## 16. Phase 2 results (20 held-out seeds 101-120, 30 days each, day 0 excluded)

Run: `scripts/run_phase2.py --days 30 --seeds 20`. Building the plans takes 8.5 min and the twin runs take 3 min. Two full runs gave identical plans and identical per-seed results (compare hash `88a42dc32e1d66be`).

| Mean per day | P0 naive | P1 routing only | P2 routing + frequency + fleet |
|---|---|---|---|
| Pickups (supplier visits) | 60 | 60 | 37 |
| km | 2,307 | 2,069 | 1,555 |
| Driver shifts / overtime h | 10 / 7.5 | 10 / 8.0 | 9 / 2.9 |
| Trucks | 5 | 5 | 5 |
| Truck fill | 0.73 | 0.77 | 0.86 |
| Planned lead (dispatch to supermarket) | 246 min | 243 min | 211 min |
| Logistics cost (twin) | $9,898 | $9,681 | $8,796 |
| Line-down | 9.1 min | 7.4 min | 3.8 min |

Paired differences vs P0 (same seeds), with 95% bootstrap intervals over seeds:

| | P1 routing only | P2 routing + frequency + fleet |
|---|---|---|
| Logistics $/day | -$217 [-226, -207] (-2.2%) | **-$1,102 [-1,109, -1,096] (-11.1%)** |
| ... transport | -$285 | -$903 |
| ... driver | +$30 | -$521 |
| ... holding (capital + space) | +$35 | +$327 |
| Line-down min/day | -1.7 [-3.8, +0.1] | **-5.3 [-8.2, -2.7]** |
| Total at $1k / $5k / $20k per line-down minute | -$1.9k / -$8.8k / -$34k (the $5k and $20k intervals cross zero) | **-$6.4k / -$27.4k / -$106k (all intervals below zero)** |

**What the numbers say.**
- **Routing alone saves little (2%).** It cuts 10% of km, but drivers are paid by the shift, and the fleet cannot shrink, so shorter routes do not remove a shift or a truck.
- **The saving is in frequency.** Ten of the 15 suppliers get fewer pickups: seven drop to two a day and three to one. Fewer, fuller visits cut km by a third and remove one driver shift. That outweighs the extra inventory: about $1,400/day saved against $327/day more holding. This is the transport-versus-inventory trade-off, in numbers.
- **Line-down also fell.** P2's routes are shorter (211 vs 246 min planned lead), its dispatches almost never run late (average delay 0.15 vs 1.7 min per trip), and it has 23 fewer pickups a day that can go wrong.
- **The fleet stays at 5 trucks in every plan.** The 04:00 and 09:00 waves each carry 5 of the 16 production hours of volume, about 120 containers against 5 x 26 = 130. The wave schedule, not routing, is the binding constraint (see below).

**Promise vs twin (figure 6).**
- **Cost.** The optimizer's deterministic planning cost overstates the twin's logistics cost by a steady 4.3-4.6% on every plan. Most of the gap is holding capital, which the planning model estimates conservatively. Because the error is the same for every plan, the ranking and the savings carry over: P2 was promised $1,139/day better than P0, and the twin measured $1,102 (97% of the promise).
- **Lead times.** Planned leads are within 1.3% of measured ones.

**What the twin caught: a trimming rule that breaks when frequencies differ.**
- **First evaluation.** P2 looked like a disaster: +27.7 line-down min/day (CI +24 to +31).
- **Diagnosis.** 33 of P2's 37 line-down minutes came from one supplier picked twice a day. The capacity-trimming rule had cut 338 containers from it over 3 seeds x 30 days.
- **Cause.** The rule trimmed whichever part had the most hours of cover after the order. Under mixed frequencies that is always the least frequent supplier, whose order has to last 10 hours.
- **Fix.** Trimming now evens out the safety hours each part has left at the end of its cycle (section 5, `test_trimming`). It applies to every plan.
- **After the fix.** P2 has less line-down than the naive plan.
- **Disclosure.** The diagnosis used seeds 101-105, which are part of the held-out set. The fix has no tunable parameter, but it was motivated by results on those seeds.

**What did not work, and is not built.**
- **Headroom (planning trucks to 85%).** This was the original setting, and it is infeasible under the current waves, for P1 and for P2 (the final solver confirms it). So the optimized plan runs trucks at 86% average fill, with no planned slack in the morning waves.
- **Next lever: wave timing.** The natural next lever is the dispatch schedule itself. Waves at 04/09/14/19 put 31% of the daily volume on each morning wave and 13% on the evening wave. Rebalancing them could free a truck and allow headroom. It is not built: it is outside the Phase 2 scope.

**Deviations from the original plan, and why.**
1. **Lead estimator check.** It is checked against the twin's mean lead, not the median. The estimator is an expected value, and the lead distribution is right-skewed. All routes are within 7% of the mean.
2. **Planning fill.** `max_fill` is 1.0, not 0.85. At 0.85, every-wave routing is infeasible (the morning waves need 120 of 130 containers), and frequency search cannot fix it.
3. **Slack.** `slack_min` is 15, not 45. The naive plan's own longest route ends 22-30 min before the next wave, so 45 would rule out the baseline.
4. **Solver.**
   - Savings construction, optional visits, greedy descent during the search and guided local search for final solves, instead of cheapest-arc + guided local search. Section 8.1 explains why; each change came from a failed test.
   - Demand is kept in tenths of a container: rounding up each supplier added about 7 containers per wave.
5. **Leg timing.** `route_timing` uses the rush factor at each leg's departure (as the twin does), not a wave average. The VRP still uses the wave average.
6. **Trimming rule** replaced (above).
7. **New module.** `evaluate.py` (twin runs and bootstrap) was added so the experiment logic is testable.
8. **Phase 1 numbers changed** with these model changes (section 15).
9. **Abandoned P3.** A headroom variant (85%, then 90%) was attempted and abandoned: 85% was infeasible. The 90% attempt stopped at the old solver's safety time limit, so it was inconclusive, and 90% was not retried after the solver fix.

## 17. Phase 3 results (held-out seeds 101-120, 15 days each, day 0 excluded)

Run: `scripts/run_phase3.py`. It tunes on seeds 1-5, tests on seeds 101-120 and sweeps buffers on seeds 101-110 (section 11). Two full runs gave identical results (hash `6231c799b73e7222`). A third run, made after the post-hoc plumbing change, reproduced the hash (`outputs/phase3_check`).

All arms run on the P2 plan with 2 h of safety stock. Costs are at $5,000 per line-down minute unless stated. Differences are paired by seed (common random numbers), with 95% bootstrap intervals. "Total" means logistics plus line-down.

### 17.1 Pre-registered result (as planned: probabilistic triggers look 2 h ahead)

Tuned thresholds: det m = 0 min; bayes and fixed tau = 0.4; the same in both regimes.

| Arm | R1 total $/day | R1 vs never | R2 total $/day | R2 vs never | Actions/day (R1 / R2) |
|---|---|---|---|---|---|
| never | 24,593 | | 30,327 | | 0 / 0 |
| periodic + value | 17,038 | -7.6k [-12.4k, -3.5k] | 20,276 | -10.1k [-13.8k, -6.4k] | 0.46 / 0.53 |
| periodic + always | 15,526 | -9.1k [-14.1k, -4.8k] | 22,287 | -8.0k [-13.6k, -1.2k] | 0.64 / 0.89 |
| event + value | 23,071 | -1.5k [-6.8k, +5.4k] | 24,244 | -6.1k [-13.2k, +4.1k] | 0.59 / 0.71 |
| event + always | 21,082 | -3.5k [-8.8k, +4.3k] | 26,998 | -3.3k [-10.3k, +5.0k] | 1.16 / 1.57 |
| det + value | 15,056 | -9.5k [-14.6k, -5.3k] | 19,247 | -11.1k [-16.3k, -5.6k] | 0.58 / 0.75 |
| **det + always** | **11,467** | **-13.1k [-20.4k, -7.1k]** | **15,514** | **-14.8k [-20.3k, -9.5k]** | 0.90 / 1.32 |
| bayes + value | 21,413 | -3.2k [-11.6k, +8.1k] | 23,143 | -7.2k [-13.2k, +1.3k] | 0.38 / 0.46 |
| bayes + always | 17,400 | -7.2k [-10.5k, -4.0k] | 21,015 | -9.3k [-13.9k, -4.2k] | 0.64 / 0.71 |
| fixed + value | 20,771 | -3.8k [-12.1k, +7.4k] | 23,588 | -6.7k [-12.7k, +1.6k] | 0.40 / 0.45 |

| Hypothesis | R1 | R2 | Verdict |
|---|---|---|---|
| H1: in R1, bayes + value ≈ det + value (CI includes 0) | +6.4k [-2.6k, +20.6k] | | Consistent, but the interval is wide and the point estimate favours det |
| H2: in R2, bayes + value < det + value | | +3.9k [-4.1k, +14.6k] | **Not supported** |
| H3: the value gate cuts actions (bayes + value − bayes + always) | -0.27 [-0.62, -0.04] actions/day | -0.25 [-0.43, -0.11] actions/day | Yes, by 35-42% |
| H3: ... at equal or lower cost | +4.0k [-2.1k, +13.0k] | +2.1k [-0.4k, +6.1k] | Not supported: the point estimate is higher cost in both regimes |
| H4: dynamic policies move the cheapest buffer left | never 4 h; det and bayes 3 h | all three at 4 h, the edge of the grid | R1 yes; R2 cannot tell |

**What the numbers say.**
- **Every dynamic policy beat "never" on the point estimate, and the plainest one won.** Det + always acts whenever the point projection says some part will run short within 6 h. It halved total cost in both regimes, and its interval is clear of zero at $1k, $5k and $20k per minute (R1: -2.6k / -13.1k / -52.8k; R2: -2.9k / -14.8k / -59.7k). It adds $83-134/day of logistics cost.
- **Trucks are almost never re-routed.** Executed re-routes stay at or below 0.02 per day in every arm. Couriers and hot runs do the work. The likely reason is that P2's routes are short (37 pickups a day across 4 waves), which leaves little stop order to change. Material at the line is the lever, which is the point of modeling the plant (section 2).
- **The value gate cuts actions but raises cost.** All executed actions together add $30-170/day of premium freight and disruption, while one line-down minute costs $5,000. A gate that saves an action and lets a few extra minutes of stoppage through loses money. Det + value against det + always shows the same in both regimes: +3.6k [+1.1k, +6.8k] (R1) and +3.7k [+1.0k, +7.2k] (R2).
- **Buffers make most of this unnecessary** (figure 10). At its own cheapest buffer, dynamic re-planning saves $202-211/day in R1 and $39-42/day in R2. At lean buffers it is worth a lot: at 0.5-1 h, det + value saves $47k-109k/day. Dynamic re-planning substitutes for safety stock, and it pays mainly where buffers are kept lean on purpose.
- **The filter learns little on sparse corridors** (figure 7). A corridor gets only a few truck legs a day, so the Kalman filter sees an incident too late to track it. The fixed arm (one pooled variance, never updated) matches bayes in both regimes: bayes + value − fixed + value = +0.6k [-0.0k, +1.9k] (R1) and -0.4k [-1.6k, +0.4k] (R2). This comparison was computed after the run (section 17.4).

### 17.2 A confound in the design, found on review after the run

- **Different lookaheads.** The det trigger looks over the predictor's full 6 h horizon (`horizon_h`). The bayes and fixed triggers ask for P(line stops within `trigger_h` = 2 h).
- **Why that matters.** A courier takes roughly 0.5 to 3 h door to door: 10-65 km straight-line, × 1.3 for the road, each way at 60 km/h before the rush factor, plus loading, any supplier delay and 10 min of express receiving. A 2 h trigger often fires when the courier, the strongest action, is already too late.
- **Consequence.** H1 and H2 compared two things at once: point estimates versus distributions, and 6 h versus 2 h of warning. Section 17.1 cannot tell which one caused the result.
- **`trigger_h` also sets which parts count as "at risk"** in an assessment, for every policy except det: a part is at risk if P(short within `trigger_h`) > 0 (`Controller.at_risk`). Changing it therefore changes the periodic and event arms too. The det arms do not use it, and their results are identical in both runs, which checks that the change touched only what it should.

### 17.3 Post-hoc run: probabilistic triggers also look 6 h ahead

**Chosen after seeing section 17.1. It is a post-hoc analysis, and is reported next to the pre-registered result, not instead of it.**

Run: `scripts/run_phase3.py --trigger-h 6`, writing `outputs/phase3_lookahead6` (hash `3d723c9903f160ab`). It used the same seeds and the same tuning procedure. Re-tuned thresholds: det m = 0; bayes and fixed tau = 0.4 (R1) and 0.2 (R2).

**Predictor calibration at 6 h** (`scripts/validate_phase3.py --gate 2 --trigger-h 6`, tuning seeds 1-5 × 10 days, under "never"):

| Regime | Calibration slope | Brier | Climatology | Gate (0.7-1.3) |
|---|---|---|---|---|
| R1 | 0.62 | 0.0103 | 0.0394 | **fails** |
| R2 | 1.21 | 0.0088 | 0.0441 | passes |

In R1 the 6 h predictor over-predicts in the middle: checks that said 0.23 were followed by a stop 9% of the time, and checks that said 0.63 by a stop 37% of the time. When it said ≥ 0.9, the line stopped every time (143/143 in R1, 151/151 in R2). R1 results below inherit this caveat.

| Arm | R1 total $/day | R1 vs never | R2 total $/day | R2 vs never | Actions/day (R1 / R2) |
|---|---|---|---|---|---|
| never | 24,593 | | 30,327 | | 0 / 0 |
| det + value (unchanged) | 15,056 | -9.5k [-14.6k, -5.3k] | 19,247 | -11.1k [-16.3k, -5.6k] | 0.58 / 0.75 |
| det + always (unchanged) | 11,467 | -13.1k [-20.4k, -7.1k] | 15,514 | -14.8k [-20.3k, -9.5k] | 0.90 / 1.32 |
| bayes + value | 13,130 | -11.5k [-18.8k, -6.0k] | 12,333 | -18.0k [-24.4k, -12.2k] | 0.62 / 1.55 |
| **bayes + always** | **11,462** | **-13.1k [-21.0k, -7.0k]** | **10,169** | **-20.2k [-27.5k, -13.4k]** | 0.90 / 1.85 |
| fixed + value | 12,116 | -12.5k [-21.9k, -6.0k] | 12,243 | -18.1k [-24.7k, -12.2k] | 0.61 / 1.58 |
| periodic + value | 21,807 | -2.8k [-15.5k, +9.3k] | 51,535 | **+21.2k [+8.9k, +32.9k]** | 9.6 / 10.1 |
| periodic + always | 20,382 | -4.2k [-12.0k, +3.4k] | 31,838 | +1.5k [-7.0k, +9.7k] | 15.4 / 16.3 |
| event + value | 36,262 | **+11.7k [+1.5k, +21.7k]** | 61,812 | **+31.5k [+10.8k, +54.2k]** | 18.4 / 19.0 |
| event + always | 17,945 | -6.6k [-16.6k, +1.3k] | 20,697 | -9.6k [-15.9k, -3.6k] | 33.2 / 34.9 |

| Comparison (total $/day) | R1 | R2 |
|---|---|---|
| bayes + value − det + value (the H1/H2 test) | -1.9k [-4.7k, -0.2k] | **-6.9k [-12.8k, -2.0k]** |
| bayes + always − det + always | -0.0k [-1.1k, +1.5k] | **-5.3k [-9.6k, -1.8k]** |
| bayes + value − fixed + value (the value of updating) | +1.0k [-0.0k, +3.0k] | +0.1k [-0.2k, +0.5k] |
| bayes + value − bayes + always (the gate) | +1.7k [+0.1k, +3.5k] | +2.2k [+0.2k, +5.4k] |
| Cheapest buffer (figure 10) | never 4 h; det 3 h; bayes 3 h | never 4 h; det 4 h; **bayes 3 h** |

**What the numbers say.**
- **With equal lookahead, the probabilistic trigger wins in the drifting regime under both gates.** In R1 it ties under "always" (as H1 predicted) and wins by $1.9k under "value", which comes with the R1 calibration caveat. The 17.1 result came mostly from the lookahead mismatch, not from the method.
- **The gain comes from modeling uncertainty, not from the Kalman updating.** The fixed arm, with one pooled variance that never updates, does as well as bayes in both regimes. This agrees with figure 7. The honest name for the winning idea is "act on the probability of a stop, looking far enough ahead", not "Bayesian updating".
- **The value gate still does not pay.** It cuts about 0.3 actions/day and costs $1.7k-2.2k/day more than acting every time.
- **The cheapest buffer moves left in R2.** Bayes + value reaches its cheapest buffer at 3 h ($8,981/day), where det and never need 4 h ($9,099 and $9,141). Even so, the saving at the cheapest buffer is $118-160/day. At lean buffers the gap is large: at 0.5 h, bayes + value costs $67k/day against $133k for det + value and $233k for never.
- **A warning about the value gate's scoring.** With a 6 h window, every part with any chance of running short in 6 h counts as at risk. The periodic and event arms then assess 48-100 times a day and act 10-35 times. Their logistics cost rises by $1.9k-6.9k/day.
  - Under the value gate, which executes only actions the predictor scores as cheaper, they also had more line-down than "never" in R2: +3.8 [+1.4, +6.2] min/day (periodic) and +5.4 [+1.3, +10.0] min/day (event).
  - So the predictor's ranking of candidate actions is not trustworthy when it is asked this often over a 6 h window.
  - This is **not diagnosed.** One possible cause, not checked: the predictor scores only the next 6 h, so an action that delays another supplier's load past that window looks free (section 13).
  - The risk-triggered arms assess only 1-2 times a day, which limits their exposure. The cause still has to be found before the value gate is trusted in any arm.

### 17.4 How the extra comparisons were computed

The comparisons not in the pre-registered list were computed after the run, from `test.csv`, with the same paired bootstrap (`evaluate.bootstrap_ci`, seed 0, 10,000 resamples). They are exploratory: bayes − fixed, bayes + always − det + always, and det + value − det + always. `hypotheses.csv` and the run hash contain only the comparisons the script makes.

### 17.5 Phase 3 in short

1. Pre-registered: the simple stockout rule won, and the Bayesian hypothesis (H2) was not supported.
2. A review after the run found a lookahead confound in the design.
3. Post hoc, with equal lookahead: a probabilistic trigger beat the simple rule by $5-7k/day in the drifting regime. The gain comes from treating ETAs as distributions, not from the filter's updating.
4. In every analysis:
   - Material at the line, through couriers and hot runs, is the lever, not truck re-routing.
   - An "only act if it pays" gate loses money when stoppage costs this much.
   - A bigger buffer makes most dynamic re-planning unnecessary. Dynamic re-planning earns its keep when the plant runs lean on purpose.

### 17.6 Deviations from the original plan, and why

1. **Gate 2 is scored at the line level.** The per-part outcome is censored: the twin records only the parts short when the line stops (section 10).
2. **Det margin grid.** The grid now includes -30 and -15 min. Point projections use mean, late-skewed durations, so the best margin could be negative. The tuned margin was 0 in every run.
3. **New action: leave a not-ready supplier now.** The original action list had no way to cut a supplier wait short. The action interrupts the wait, and the dropped order goes to the supplier's next pickup (section 8.2). It has tests on both the event path and the interrupt path.
4. **Two bugs, found and fixed before the experiment.**
   - The predictor dropped trips that were overdue but not started. Fixed with a trips-started count in the Observation.
   - The event trigger read a NaN leg time. Fixed with an "arrived" truck status.
5. **The post-hoc lookahead run** (sections 17.2 and 17.3). It added `--trigger-h` to `run_phase3.py` and `validate_phase3.py`.
6. **Figures.** After the runs, `run_phase3.py --figures-only` was added to re-render figures 7-11 from the saved CSVs.
   - Figure 8 now puts the checks per bin in their own panel: both regimes' labels collided at (0, 0).
   - Figure 10 now has a log cost axis and lists each policy's cheapest buffer.
   - No result changed.
7. Not built from the plan: line-speed (JPH) events and class-based coverage.

## 18. Stress tests (`src/inbound_twin/stress.py`, `scripts/stress_test.py`)

Each stress test is one disruption in one window of one day (`StressCfg`, off by default, so no earlier result changes). Runs use R1 (normal traffic), 8 days, the disruption on day 7, and the same held-out seeds 101-120. Each seed runs under the fixed plan (never re-plan) and under the Shortage Early Warning (det + always, section 8.2), paired through common random numbers.

| Disruption | How it is modelled | Fixed plan: stop days, mean min down | Early warning: stop days, mean min down | Mean cost of the day |
|---|---|---|---|---|
| None | | 3 / 20, 3.7 | 0 / 20, 0.0 | $27,531 → $8,805 |
| Supplier 10 shut 08:00-16:00 | any truck or courier arriving waits until 16:00 | 20 / 20, 184.9 | 20 / 20, 112.5 | $933,276 → $573,827 (−38.5%) |
| Road to supplier 5 at a crawl, 07:00-13:00 | legs to or from supplier 5 departing in the window take 3× as long | 20 / 20, 88.9 | 14 / 20, 33.6 | $453,296 → $178,893 (−60.5%) |
| Two of four doors out, 06:00-14:00 | two doors are held at top priority from the first moment each frees up | 3 / 20, 6.7 | 0 / 20, 0.0 | $42,545 → $8,827 |

- The early warning cannot make a shut supplier ship. Its couriers wait for the same supplier, so it shortens the stop but does not prevent it. A second source is outside this model.
- A slow road is where early action pays most. The pings and filter show the trucks running late, and the couriers and hot runs land before the line runs dry.
- Half the dock for a morning costs little. The yard queue clears before any part runs out at 2 h of safety stock.
- The controller is not told about any disruption. It sees only what it always sees: pings, stock, dock waits, a supplier not ready.
- For each scenario the median day by fixed-plan cost is exported as a replay (`showcase/web/data/replay_<scenario>.json`) and summarized in `stress_summary.json` for the control tower.
