"""Phase 1 figures (static PNG). Colors follow the reference categorical order;
line-down uses the reserved 'critical' status color and is always labelled."""
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # slots 1-6, fixed order
PLAN_LABEL = {"naive": "P0 naive", "routing": "P1 routing only",
              "optimized": "P2 routing + frequency + fleet"}
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "lines.linewidth": 2,
})


def _hours_axis(ax, day: int) -> None:
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel(f"Hour of day {day}")


def _title_and_legend(fig, title: str, handles, labels, ncol: int, top: float = 0.9) -> None:
    """Title on its own line, legend in a row below it, plots underneath."""
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontweight="bold")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=ncol,
               frameon=False)


def _shade_downtime(ax, tw, day: int, label: bool) -> None:
    lo = day * 1440.0
    first = True
    for a, b, _ in tw.downtime:
        if b <= lo or a >= lo + 1440:
            continue
        ax.axvspan((max(a, lo) - lo) / 60, (min(b, lo + 1440) - lo) / 60, color=CRITICAL, alpha=0.18,
                   lw=0, label="Line down" if (label and first) else None)
        first = False


def worst_day(tw, kpis) -> int:
    """The day with the most line-down minutes (ties: the earliest)."""
    if kpis["line_down_min"].max() > 0:
        return int(kpis.loc[kpis["line_down_min"].idxmax(), "day"])
    return int(kpis["day"].iloc[0])


def parts_to_show(tw, day: int, k: int = 3) -> list[int]:
    lo, hi = day * 1440.0, (day + 1) * 1440.0
    chosen = []
    for a, _, parts in tw.downtime:
        if lo <= a < hi:
            chosen += [p for p in parts if p not in chosen]
    rate_h = np.array([p.usage for p in tw.net.parts]) * tw.cfg.line.jph
    mask = (tw.samples["t"] >= lo) & (tw.samples["t"] < hi)
    min_cover = (tw.samples["supermarket"][mask] / rate_h).min(axis=0)
    for p in np.argsort(min_cover):
        if len(chosen) >= k:
            break
        if int(p) not in chosen:
            chosen.append(int(p))
    return chosen[:k]


def fig_inventory(tw, day: int, path: Path) -> None:
    parts = parts_to_show(tw, day)
    lo = day * 1440.0
    mask = (tw.samples["t"] >= lo) & (tw.samples["t"] <= lo + 1440)
    hours = (tw.samples["t"][mask] - lo) / 60
    rate_h = np.array([p.usage for p in tw.net.parts]) * tw.cfg.line.jph
    fig, axes = plt.subplots(len(parts), 1, figsize=(9, 2.3 * len(parts) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    series = [("Supermarket", "supermarket"), ("Line-side rack", "line"),
              ("Inbound (on truck, at dock, in receiving)", "pipeline")]
    for i, (ax, p) in enumerate(zip(axes, parts)):
        for (name, key), color in zip(series, SERIES):
            ax.plot(hours, tw.samples[key][mask][:, p] / rate_h[p], color=color, label=name)
        _shade_downtime(ax, tw, day, label=(i == 0))
        part = tw.net.parts[p]
        ax.set_title(f"Part {p} (supplier {part.supplier}, {part.upc} units/container, "
                     f"{part.kanban} kanban cards)", loc="left")
        ax.set_ylabel("Hours of cover")
        ax.set_ylim(bottom=0)
    _hours_axis(axes[-1], day)
    _title_and_legend(fig, f"Where each part's stock sits during day {day} "
                           "(cover = units / consumption rate)",
                      *axes[0].get_legend_handles_labels(), ncol=4)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_gantt(tw, day: int, path: Path) -> None:
    lo = day * 1440.0
    cats = ["Driving", "Gate & waiting for door / crew", "Loading at supplier",
            "Supplier not ready", "Unloading & dock out"]
    color = dict(zip(cats, SERIES))
    bars = {c: [] for c in cats}
    road = {"travel": "Driving", "load": "Loading at supplier", "late": "Supplier not ready"}
    for truck, kind, a, b in tw.segments:
        if lo <= a < lo + 1440:
            bars[road[kind]].append((truck, a, b))
    for v in tw.visits:
        if v.kind != "milk" or not lo <= v.t_arrive < lo + 1440 or math.isnan(v.t_leave):
            continue
        bars["Gate & waiting for door / crew"].append((v.key, v.t_arrive, v.t_crew))
        bars["Unloading & dock out"].append((v.key, v.t_crew, v.t_leave))
    n = tw.cfg.fleet.n_trucks
    fig, ax = plt.subplots(figsize=(10, 0.55 * n + 1.6))
    for c in cats:
        for truck, a, b in bars[c]:
            ax.barh(truck, (b - a) / 60, left=(a - lo) / 60, height=0.55, color=color[c],
                    edgecolor=SURFACE, linewidth=1.0)
        ax.barh(np.nan, 0, color=color[c], label=c)
    ax.set_yticks(range(n), [f"Truck {i}" for i in range(n)])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _hours_axis(ax, day)
    _title_and_legend(fig, f"Milk-run trucks, day {day}: where the time goes",
                      *ax.get_legend_handles_labels(), ncol=3, top=0.82)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _step_count(intervals, lo: float, hi: float):
    events = sorted([(a, 1) for a, b in intervals if b > lo and a < hi]
                    + [(b, -1) for a, b in intervals if b > lo and a < hi])
    level = sum(1 for a, b in intervals if a <= lo < b)
    xs, ys = [0.0], [level]
    for t, dv in events:
        if t <= lo:
            continue
        level += dv
        xs.append((min(t, hi) - lo) / 60)
        ys.append(level)
    xs.append(24.0)
    ys.append(level)
    return xs, ys


def fig_dock(tw, day: int, path: Path) -> None:
    lo, hi = day * 1440.0, (day + 1) * 1440.0
    horizon = tw.days * 1440.0
    end = lambda t: horizon if math.isnan(t) else t  # noqa: E731
    doors = [(v.t_door, end(v.t_leave)) for v in tw.visits if not math.isnan(v.t_door)]
    crews = [(v.t_crew, end(v.t_unloaded)) for v in tw.visits if not math.isnan(v.t_crew)]
    yard_t = np.array([t for t, _ in tw.yard_trace])
    yard_n = np.array([q for _, q in tw.yard_trace])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for (name, iv, cap), color in zip([("Doors occupied", doors, tw.cfg.plant.n_doors),
                                       ("Crews unloading", crews, tw.cfg.plant.n_crews)], SERIES):
        xs, ys = _step_count(iv, lo, hi)
        ax1.step(xs, ys, where="post", color=color, label=f"{name} (of {cap})")
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, tw.cfg.plant.n_doors + 0.5)
    ax1.set_yticks(range(tw.cfg.plant.n_doors + 1))
    for v in tw.visits:
        if v.kind == "milk" and lo <= v.t_arrive < hi:
            ax1.plot((v.t_arrive - lo) / 60, tw.cfg.plant.n_doors + 0.25, marker="v", ms=8,
                     color=INK_2, ls="none")
    ax1.plot([], [], marker="v", ms=8, color=INK_2, ls="none", label="Milk-run truck arrives")

    # A truck that gets a door the moment it clears the gate joins and leaves the
    # yard at the same instant; keep only the last level at each time so that
    # zero-length visits don't draw as spikes.
    last = {}
    for t, q in zip(yard_t, yard_n):
        last[float(t)] = int(q)
    ts = np.array(sorted(last))
    qs = np.array([last[t] for t in ts])
    in_day = (ts >= lo) & (ts < hi)
    start_level = qs[ts < lo][-1] if (ts < lo).any() else 0
    end_level = qs[in_day][-1] if in_day.any() else start_level
    xs = np.concatenate([[0.0], (ts[in_day] - lo) / 60, [24.0]])
    ys = np.concatenate([[start_level], qs[in_day], [end_level]])
    ax2.step(xs, ys, where="post", color=SERIES[0])
    ax2.set_ylabel("Trucks in yard")
    ax2.set_ylim(0, max(2, int(ys.max()) + 1))
    ax2.set_yticks(range(int(ax2.get_ylim()[1]) + 1))
    _hours_axis(ax2, day)
    _title_and_legend(fig, f"Dock, day {day}: doors, crews and the yard queue (all carriers)",
                      *ax1.get_legend_handles_labels(), ncol=3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Phase 2
def fig_routes(net, cfg, plans, wave: int, path: Path) -> None:
    """One panel per plan: the routes of one dispatch wave on the map."""
    from .plan import wave_index
    fig, axes = plt.subplots(1, len(plans), figsize=(4.6 * len(plans), 5.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    xy = {s.id: (s.x_km, s.y_km) for s in net.suppliers}
    xy[0] = (0.0, 0.0)
    trucks_seen = set()
    for ax, plan in zip(axes, plans):
        routes = [r for r in plan.routes if wave_index(cfg, r.dispatch_min) == wave]
        visited = {s for r in routes for s in r.suppliers}
        for r in routes:
            pts = np.array([xy[n] for n in (0, *r.suppliers, 0)])
            ax.plot(pts[:, 0], pts[:, 1], color=SERIES[r.truck % len(SERIES)], lw=2,
                    label=f"Truck {r.truck}" if r.truck not in trucks_seen else None, zorder=2)
            trucks_seen.add(r.truck)
        for s in net.suppliers:
            hit = s.id in visited
            ax.plot(s.x_km, s.y_km, "o", ms=8, mfc=INK_2 if hit else SURFACE, mec=INK_2, mew=1.5,
                    zorder=3)
            ax.annotate(str(s.id), (s.x_km, s.y_km), xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color=INK_2)
        ax.plot(0, 0, "s", ms=11, color=INK, zorder=4)
        ax.set_title(f"{PLAN_LABEL.get(plan.name, plan.name)}\n{len(routes)} trucks, "
                     f"{len(visited)} of {len(net.suppliers)} suppliers this wave", loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel("km east of plant")
    axes[0].set_ylabel("km north of plant")
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    handles = [handles[i] for i in order] + [
        plt.Line2D([], [], marker="s", color=INK, ls="none", ms=9),
        plt.Line2D([], [], marker="o", mfc=SURFACE, mec=INK_2, ls="none", ms=8)]
    labels = [labels[i] for i in order] + ["Plant", "Not picked up this wave"]
    h = cfg.fleet.dispatch_h[wave]
    _title_and_legend(fig, f"Milk-run routes for the {int(h):02d}:00 wave", handles, labels,
                      ncol=len(labels), top=0.86)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_costs(sim, cmp, path: Path) -> None:
    """Left: logistics cost by term per plan. Right: line-down minutes with 95 % CIs."""
    from .evaluate import LOGISTICS, bootstrap_ci
    names = list(dict.fromkeys(sim.plan))
    mean = sim.groupby("plan", sort=False).mean(numeric_only=True).loc[names]
    labels = [PLAN_LABEL.get(n, n) for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2 + 0.35 * len(names)),
                                   gridspec_kw={"width_ratios": [1.6, 1]}, sharey=True)
    y = np.arange(len(names))
    left = np.zeros(len(names))
    pretty = {"transport": "Transport", "driver": "Driver shifts", "trucks": "Truck lease",
              "handling": "Handling", "holding_capital": "Holding (capital)",
              "holding_space": "Holding (space)"}
    for term, color in zip(LOGISTICS, SERIES):
        ax1.barh(y, mean[term], left=left, height=0.5, color=color, edgecolor=SURFACE, lw=1.5,
                 label=pretty[term])
        left += mean[term].values
    for yi, total in zip(y, left):
        ax1.annotate(f"${total:,.0f}", (total, yi), xytext=(6, 0), textcoords="offset points",
                     va="center", color=INK, fontsize=9)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Logistics cost, USD per day (mean of seeds)")
    ax1.set_xlim(0, left.max() * 1.18)
    ax1.grid(axis="y", visible=False)
    for yi, n in zip(y, names):
        vals = sim.loc[sim.plan == n, "line_down_min"].values
        lo, hi = bootstrap_ci(vals)
        ax2.errorbar(vals.mean(), yi, xerr=[[vals.mean() - lo], [hi - vals.mean()]], fmt="o", ms=8,
                     color=SERIES[0], ecolor=INK_2, capsize=4, lw=1.5)
        ax2.annotate(f"{vals.mean():.1f}", (hi, yi), xytext=(6, 0), textcoords="offset points",
                     va="center", color=INK, fontsize=9)
    ax2.set_xlabel("Line-down minutes per day (95% CI over seeds)")
    ax2.set_xlim(left=0)
    ax2.grid(axis="y", visible=False)
    _title_and_legend(fig, "What each plan costs in the twin (same seeds for every plan)",
                      *ax1.get_legend_handles_labels(), ncol=6, top=0.82)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_promise(sim, path: Path) -> None:
    """The optimizer's deterministic promise next to what the twin measured."""
    names = list(dict.fromkeys(sim.plan))
    mean = sim.groupby("plan", sort=False).mean(numeric_only=True).loc[names]
    labels = [PLAN_LABEL.get(n, n) for n in names]
    y = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 2.8 + 0.35 * len(names)), sharey=True)
    for ax, planned, measured, unit in [(ax1, "planned_total", "logistics", "Logistics cost, USD/day"),
                                        (ax2, "lead_planned_min", "lead_measured_min",
                                         "Dispatch to supermarket, minutes")]:
        for yi, n in enumerate(names):
            a, b = mean.loc[n, planned], mean.loc[n, measured]
            ax.plot([a, b], [yi, yi], color=GRID, lw=3, zorder=1)
            ax.annotate(f"{(b - a) / a:+.1%}", (max(a, b), yi), xytext=(8, 0),
                        textcoords="offset points", va="center", color=INK_2, fontsize=9)
        ax.plot(mean[planned], y, "o", ms=9, mfc=SURFACE, mec=SERIES[0], mew=2, ls="none",
                label="Planned (optimizer's deterministic model)", zorder=2)
        ax.plot(mean[measured], y, "o", ms=9, color=SERIES[0], ls="none",
                label="Twin (mean of seeds)", zorder=3)
        ax.set_xlabel(unit)
        ax.grid(axis="y", visible=False)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + (hi - lo) * 0.15)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    _title_and_legend(fig, "Promise vs twin: what the optimizer expected and what the twin measured",
                      *ax1.get_legend_handles_labels(), ncol=2, top=0.8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Phase 3
POLICY_ORDER = ("never", "periodic+always", "periodic+value", "event+always", "event+value")


def _policy_sort_key(name: str):
    base = name.split("(")[0] + "+" + name.split("+")[-1] if "(" in name else name
    order = ["never", "periodic+always", "periodic+value", "event+always", "event+value",
             "det+always", "det+value", "fixed+value", "bayes+always", "bayes+value"]
    return order.index(base) if base in order else len(order)


def fig_corridor(rows, day: int, supplier: int, path: Path) -> None:
    lo, hi = day * 1440.0, (day + 1) * 1440.0
    r = np.array([x for x in rows if lo <= x[0] < hi])
    if r.size == 0:
        return
    h = (r[:, 0] - lo) / 60
    fig, ax = plt.subplots(figsize=(10, 3.8))
    band = 1.2816 * r[:, 2]
    ax.fill_between(h, np.exp(r[:, 1] - band), np.exp(r[:, 1] + band), color=SERIES[0], alpha=0.18, lw=0,
                    label="Filter 80% band")
    ax.plot(h, np.exp(r[:, 1]), color=SERIES[0], label="Filter estimate")
    ax.step(h, np.exp(r[:, 3]), where="post", color=SERIES[1], label="True congestion (hidden from the filter)")
    ax.set_ylabel("Travel-time multiplier")
    _hours_axis(ax, day)
    _title_and_legend(fig, f"Corridor of supplier {supplier}, day {day} (drifting regime): "
                           "what the Kalman filter believes vs the truth", *ax.get_legend_handles_labels(), ncol=3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_reliability(rel: dict, horizon_h: float, path: Path) -> None:
    """Top: observed vs predicted per bin. Bottom: checks per bin. The counts get their own
    panel because both regimes put thousands of checks at (0, 0) and their labels collided."""
    bins = [(0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)]
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(6.2, 7.0), gridspec_kw={"height_ratios": [3, 1.4]})
    ax.plot([0, 1], [0, 1], color=GRID, lw=2, zorder=1)
    width = 0.8 / len(rel)
    for i, ((name, (p, y)), color) in enumerate(zip(rel.items(), SERIES)):
        xs, ys, ns = [], [], []
        for lo, hi in bins:
            sel = (p >= lo) & (p < hi)
            ns.append(int(sel.sum()))
            if sel.sum() >= 3:
                xs.append(p[sel].mean())
                ys.append(y[sel].mean())
        ax.plot(xs, ys, "o-", color=color, ms=8, label=f"{name} regime", zorder=3)
        pos = np.arange(len(bins)) + (i - (len(rel) - 1) / 2) * width
        axn.bar(pos, ns, width=width, color=color)
        for x, n in zip(pos, ns):
            if n:
                axn.annotate(f"{n:,}", (x, n), xytext=(0, 2), textcoords="offset points", ha="center",
                             va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(f"Predicted P(line stops within {horizon_h:g} h), mean of the bin")
    ax.set_ylabel("Observed share that stopped")
    axn.set_yscale("log")
    axn.set_ylim(0.7, max(len(p) for p, _ in rel.values()) * 6)
    axn.set_xticks(range(len(bins)), [f"{lo:g}-{min(hi, 1):g}" for lo, hi in bins])
    axn.set_xlabel("Predicted probability bin")
    axn.set_ylabel("Checks (log)")
    axn.grid(axis="x", visible=False)
    _title_and_legend(fig, f"Predictor reliability, {horizon_h:g} h ahead (tuning seeds, checks every 15 min)",
                      *ax.get_legend_handles_labels(), ncol=2, top=0.9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_arms(test, rate: float, path: Path, note: str = "") -> None:
    from .evaluate import bootstrap_ci
    regimes = list(dict.fromkeys(test.regime))
    fig, axes = plt.subplots(len(regimes), 2, figsize=(12, 3.4 * len(regimes)),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    axes = np.atleast_2d(axes)
    for row, r in zip(axes, regimes):
        df = test[test.regime == r].assign(total=lambda d: d.logistics + rate * d.line_down_min)
        pols = sorted(df.policy.unique(), key=_policy_sort_key)
        y = np.arange(len(pols))
        for yi, pol in zip(y, pols):
            v = df[df.policy == pol]
            lo, hi = bootstrap_ci(v.total.values)
            color = INK_2 if pol == "never" else SERIES[0]
            row[0].errorbar(v.total.mean(), yi, xerr=[[v.total.mean() - lo], [hi - v.total.mean()]], fmt="o",
                            ms=7, color=color, ecolor=GRID, capsize=3, lw=2)
            row[1].barh(yi, v.executed_per_day.mean(), height=0.55, color=SERIES[2])
        row[0].set_yticks(y, pols)
        row[0].invert_yaxis()
        row[1].set_yticks(y, [""] * len(pols))
        row[1].invert_yaxis()
        row[0].set_title(f"{r}: total cost per day at ${rate:,.0f}/line-down min (95% CI)", loc="left")
        row[1].set_title("Executed actions per day", loc="left")
        for ax in row:
            ax.grid(axis="y", visible=False)
    _title_and_legend(fig, f"Re-planning policies on held-out seeds 101-120 (P2 plan){note}", [], [], ncol=1,
                      top=0.95)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_ucurves(curves, rate: float, path: Path, note: str = "") -> None:
    """Log y-axis: the curves span $9k to $230k, and on a linear axis everything from 2 h up
    sat on the floor. Rings of different sizes so that minima at the same point stay visible;
    the minima are also written out, since they differ by only a few percent."""
    regimes = list(dict.fromkeys(curves.regime))
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.6 * len(regimes), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, regimes):
        df = curves[curves.regime == r].assign(total=lambda d: d.logistics + rate * d.line_down_min)
        g = df.groupby(["policy", "ss_h"]).total.mean().reset_index()
        mins = []
        pols = sorted(g.policy.unique(), key=_policy_sort_key)
        for k, (pol, color) in enumerate(zip(pols, [INK_2, SERIES[1], SERIES[0]])):
            sub = g[g.policy == pol].sort_values("ss_h")
            ax.plot(sub.ss_h, sub.total, "o-", color=color, label=pol)
            i = sub.total.idxmin()
            ax.plot(sub.loc[i, "ss_h"], sub.loc[i, "total"], "o", ms=12 + 5 * k, mfc="none", mec=color, mew=2)
            mins.append(f"{pol}: {sub.loc[i, 'ss_h']:g} h, ${sub.loc[i, 'total'] / 1000:.2f}k")
        ax.text(0.98, 0.97, "Cheapest buffer\n" + "\n".join(mins), transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=INK_2, linespacing=1.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Safety stock (production hours)")
        ax.set_title(f"{r} regime", loc="left")
    axes[0].set_ylabel("Total cost per day (log scale)")
    # Thresholds can differ by regime, so the shared legend names the policy family only;
    # each panel's text box gives the tuned threshold.
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [re.sub(r"\(.*?\)", "", lab) for lab in labels]
    _title_and_legend(fig, f"Total cost vs buffer size (seeds 101-110, ${rate:,.0f}/line-down min){note}; "
                           "ring = cheapest buffer", handles, labels, ncol=3, top=0.84)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_trace(never, dyn, ctl, day: int, spec, horizon_h: float, path: Path) -> None:
    lo = day * 1440.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    tr = np.array([x for x in ctl.trace if lo <= x[0] < lo + 1440])
    if tr.size:
        ax1.plot((tr[:, 0] - lo) / 60, tr[:, 1], color=SERIES[0], label=f"P(line stops within {horizon_h:g} h)")
    ax1.axhline(spec.threshold, color=INK_2, lw=1, ls="--", label=f"trigger tau = {spec.threshold:g}")
    for dcs in ctl.decisions:
        if lo <= dcs["t"] < lo + 1440 and dcs["executed"]:
            ax1.plot((dcs["t"] - lo) / 60, 1.04, "v", color=SERIES[2], ms=9)
            ax1.annotate(dcs["label"].replace("truck ", "T"), ((dcs["t"] - lo) / 60, 1.06), rotation=60,
                         fontsize=7, color=INK_2, ha="left", va="bottom")
    ax1.plot([], [], "v", color=SERIES[2], ms=9, ls="none", label="executed action")
    ax1.set_ylim(-0.02, 1.25)
    ax1.set_ylabel("Probability")
    for yrow, tw, name in ((1, never, "never re-plan"), (0, dyn, spec.name)):
        for a, b, _ in tw.downtime:
            if b > lo and a < lo + 1440:
                ax2.barh(yrow, (min(b, lo + 1440) - max(a, lo)) / 60, left=(max(a, lo) - lo) / 60, height=0.5,
                         color=CRITICAL)
    ax2.set_yticks([0, 1], [spec.name, "never re-plan"])
    ax2.set_ylim(-0.6, 1.6)
    ax2.set_title("Line stopped", loc="left")
    ax2.grid(axis="y", visible=False)
    _hours_axis(ax2, day)
    _title_and_legend(fig, f"Day {day}, drifting regime: the trigger, the gate's actions, and the line",
                      *ax1.get_legend_handles_labels(), ncol=3, top=0.86)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
