// The recorded video's script for the hero day (R2, seed 115, day 10): one continuous day,
// no cuts; the clock slows where something happens. "Shortage Early Warning" is the
// re-planning policy (det + always, SPEC 8.2) running on the twin. Numbers in the captions
// are this replay's: truck 4 carries part 19; the first warning is at 12:15; the fixed plan's
// line stops 15:10-16:15 (64 min, $321,000); the warning's actions cost $1,116 that day.
const FPS = 30;
const DURATION = 50;

// [video second, simulated minute]; linear in between
// (the plan run's line stops at 15:10:48 = minute 910.8, just after 27 s)
const CLOCK = [[0, 520], [5, 560], [13, 725], [14.5, 735], [17, 736], [22, 830], [26, 903], [27, 911], [29.8, 917], [32.5, 976], [34, 980]];

// [from s, to s, caption, sub-line]
const CAPTIONS = [
  [0.3, 3.6, 'A digital twin of a lean plant’s inbound supply chain', 'Local suppliers, milk runs 1–4 times a day, about 2 hours of each part at the line: where one late truck can stop a line'],
  [3.6, 6.5, 'The same day, run twice', 'Top: the fixed plan. Bottom: the fixed plan plus a Shortage Early Warning that forecasts run-outs and acts.'],
  [6.5, 13.3, 'Traffic builds west of the plant', 'Truck 4 is carrying part 19 for the line. Its trip is running hours late.'],
  [13.4, 17.2, '12:15: the Early Warning flags part 19', 'It forecasts a run-out at the line about 3 hours ahead. The fixed plan sees nothing.'],
  [17.2, 22.0, 'It acts: a courier to supplier 10 and a rush to the line', 'Two couriers at $383 each. The fixed plan carries on unchanged.'],
  [22.0, 26.8, 'Part 19 at the line: running down on top, topped up below', ''],
  [26.9, 29.8, '15:10: under the fixed plan, the line stops', 'Part 19 ran out. Every minute stopped costs $5,000.'],
  [29.8, 34.0, 'With the Early Warning, the line keeps running', '64 minutes of downtime avoided ($321,000) for $1,116 of actions'],
];

// [from s, to s, spotlight rings, sidebar cards to highlight]
const FOCUS = [
  [6.5, 13.3, ['truck'], []],
  [13.4, 17.2, ['twinRack'], ['chartCard', 'alertCard']],
  [17.2, 22.0, ['courier', 'supplier'], ['alertCard']],
  [22.0, 26.9, ['planRack', 'twinRack'], ['chartCard']],
  [26.9, 29.8, ['planLine', 'planRack'], ['lineCard', 'costCard']],
  [29.8, 34.0, ['twinLine', 'twinRack'], ['lineCard', 'costCard']],
];
const RESULTS = [34.0, 44.0];
const END = [44.0, DURATION];

function simTime(sec) {
  if (sec <= CLOCK[0][0]) return CLOCK[0][1];
  for (let i = 1; i < CLOCK.length; i++) {
    const [s0, t0] = CLOCK[i - 1], [s1, t1] = CLOCK[i];
    if (sec <= s1) return t0 + (t1 - t0) * (sec - s0) / (s1 - s0);
  }
  return CLOCK[CLOCK.length - 1][1];
}
function rate(sec) {
  for (let i = 1; i < CLOCK.length; i++) {
    const [s0, t0] = CLOCK[i - 1], [s1, t1] = CLOCK[i];
    if (sec <= s1) return (t1 - t0) / (s1 - s0);
  }
  return 0;
}
const fade = (sec, a, b, d = 0.3) => Math.max(0, Math.min(1, (sec - a) / d, (b - sec) / d));

export const STORY = {
  fps: FPS,
  frames: () => Math.round(DURATION * FPS),
  at(sec) {
    const cap = CAPTIONS.find(([a, b]) => sec >= a && sec < b);
    const foc = FOCUS.find(([a, b]) => sec >= a && sec < b);
    const r = rate(sec);
    return {
      sec, t: simTime(sec),
      caption: cap ? cap[2] : null, sub: cap ? cap[3] : '', capAlpha: cap ? fade(sec, cap[0], cap[1]) : 0,
      spots: foc ? foc[2] : [], glow: foc ? foc[3] : [],
      results: sec >= RESULTS[0] && sec < RESULTS[1] ? Math.min(1, (sec - RESULTS[0]) / 1.6) : null,
      endcard: sec >= END[0] ? Math.min(1, (sec - END[0]) / 1.0) : null,
      speed: r >= 1 ? `▶ ${Math.round(r)} min / s` : '❚❚ slowed down',
    };
  },
};
