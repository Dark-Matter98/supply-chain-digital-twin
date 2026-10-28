// The recorded video's script. Each story is one continuous day, no cuts; the clock slows
// where something happens. "Shortage Early Warning" is the re-planning policy (det + always,
// SPEC 8.2) running on the twin. Numbers in the captions are the replays' own.
//
// Hero day (R2, seed 115, day 10): truck 4 carries part 19; the first warning is at 12:15; the
// fixed plan's line stops 15:10-16:15 (64 min, $321,000); the warning's actions cost $1,116.
//
// Stress test (road_closure: R1, seed 119, day 7, the median day of the 20): legs to and from
// supplier 5 take 3x as long 07:00-13:00. Truck 4 runs plant -> S5 -> S10 (parts 18 and 19)
// on that road and reaches S10 at 14:32. The warning acts at 13:00; the fixed plan's line stops
// 15:16-16:45 (88 min, part 18); with the warning it stops 15:55-16:14 (19 min, part 9). Over
// the 20 days: stopped on 20 vs 14 days, cost per day -60.5% (stress_summary.json).
const FPS = 30;

function makeStory({ duration, clock, captions, focus, results = null, end = null }) {
  function simTime(sec) {
    if (sec <= clock[0][0]) return clock[0][1];
    for (let i = 1; i < clock.length; i++) {
      const [s0, t0] = clock[i - 1], [s1, t1] = clock[i];
      if (sec <= s1) return t0 + (t1 - t0) * (sec - s0) / (s1 - s0);
    }
    return clock[clock.length - 1][1];
  }
  function rate(sec) {
    for (let i = 1; i < clock.length; i++) {
      const [s0, t0] = clock[i - 1], [s1, t1] = clock[i];
      if (sec <= s1) return (t1 - t0) / (s1 - s0);
    }
    return 0;
  }
  const fade = (sec, a, b, d = 0.3) => Math.max(0, Math.min(1, (sec - a) / d, (b - sec) / d));
  return {
    fps: FPS,
    frames: () => Math.round(duration * FPS),
    at(sec) {
      const cap = captions.find(([a, b]) => sec >= a && sec < b);
      const foc = focus.find(([a, b]) => sec >= a && sec < b);
      const r = rate(sec);
      return {
        sec, t: simTime(sec),
        caption: cap ? cap[2] : null, sub: cap ? cap[3] : '', capAlpha: cap ? fade(sec, cap[0], cap[1]) : 0,
        spots: foc ? foc[2] : [], glow: foc ? foc[3] : [],
        results: results && sec >= results[0] && sec < results[1] ? Math.min(1, (sec - results[0]) / 1.6) : null,
        endcard: end && sec >= end[0] ? Math.min(1, (sec - end[0]) / 1.0) : null,
        speed: r >= 1 ? `▶ ${Math.round(r)} min / s` : '❚❚ slowed down',
      };
    },
  };
}

// clock: [video second, simulated minute], linear in between
// captions: [from s, to s, caption, sub-line]
// focus: [from s, to s, spotlight rings, sidebar cards to highlight]
export const STORY = makeStory({
  duration: 50,
  // (the plan run's line stops at 15:10:48 = minute 910.8, just after 27 s)
  clock: [[0, 520], [5, 560], [13, 725], [14.5, 735], [17, 736], [22, 830], [26, 903], [27, 911], [29.8, 917], [32.5, 976], [34, 980]],
  captions: [
    [0.3, 3.6, 'A digital twin of a lean plant’s inbound supply chain', 'Local suppliers, milk runs 1–4 times a day, about 2 hours of each part at the line: where one late truck can stop a line'],
    [3.6, 6.5, 'The same day, run twice', 'Top: the fixed plan. Bottom: the fixed plan plus a Shortage Early Warning that forecasts run-outs and acts.'],
    [6.5, 13.3, 'Traffic builds west of the plant', 'Truck 4 is carrying part 19 for the line. Its trip is running hours late.'],
    [13.4, 17.2, '12:15: the Early Warning flags part 19', 'It forecasts a run-out at the line about 3 hours ahead. The fixed plan sees nothing.'],
    [17.2, 22.0, 'It acts: a courier to supplier 10 and a rush to the line', 'Two couriers at $383 each. The fixed plan carries on unchanged.'],
    [22.0, 26.8, 'Part 19 at the line: running down on top, topped up below', ''],
    [26.9, 29.8, '15:10: under the fixed plan, the line stops', 'Part 19 ran out. Every minute stopped costs $5,000.'],
    [29.8, 34.0, 'With the Early Warning, the line keeps running', '64 minutes of downtime avoided ($321,000) for $1,116 of actions'],
  ],
  focus: [
    [6.5, 13.3, ['truck'], []],
    [13.4, 17.2, ['twinRack'], ['chartCard', 'alertCard']],
    [17.2, 22.0, ['courier', 'supplier'], ['alertCard']],
    [22.0, 26.9, ['planRack', 'twinRack'], ['chartCard']],
    [26.9, 29.8, ['planLine', 'planRack'], ['lineCard', 'costCard']],
    [29.8, 34.0, ['twinLine', 'twinRack'], ['lineCard', 'costCard']],
  ],
  results: [34.0, 44.0],
  end: [44.0, 50],
});

export const STRESS_STORIES = {
  road_closure: makeStory({
    duration: 12.6,
    // (truck 4 leaves at 09:00; the warning acts at 13:00; the plan's line stops at 15:16:48 =
    // minute 916.8, just before 7 s)
    clock: [[0, 530], [3.2, 760], [4.0, 780], [5.6, 800], [6.6, 905], [7.0, 917], [8.6, 960], [9.8, 1008], [12.6, 1020]],
    captions: [
      [0.2, 3.4, 'Stress test: the road to supplier 5 slows to a crawl', 'Another day. From 07:00 to 13:00 every leg to or from supplier 5 takes three times as long.'],
      [3.4, 6.7, '13:00: the Early Warning sees parts 18 and 19 coming up short', 'Truck 4, which collects them at supplier 10, is still on the slow road. It sends couriers and rushes stock to the line.'],
      [6.7, 9.8, '15:16: the fixed plan’s line stops for 88 minutes', 'With the Early Warning it stops for 19 minutes, when part 9 runs short.'],
      [9.8, 12.6, 'Over 20 test days with this disruption', 'Fixed plan: the line stopped on all 20. Early Warning: on 14, and the cost per day falls 60.5%.'],
    ],
    focus: [
      [0.2, 3.4, ['truck'], ['stressCard']],
      [3.4, 6.7, ['truck', 'courier'], ['chartCard', 'alertCard']],
      [6.7, 9.8, ['planLine', 'twinLine'], ['lineCard', 'costCard']],
      [9.8, 12.6, [], ['stressCard']],
    ],
  }),
};

// The cut: [scenario ('' = the hero day), from s, to s] of each story, joined by a crossfade.
// The stress test sits between the hero day and the results card.
export const EDIT = {
  fps: FPS,
  xfade: 0.5,
  segments: [['', 0, 34.0], ['road_closure', 0, 12.6], ['', 34.4, 50]],
};
