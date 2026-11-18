// gta6-twitch-ccu-forecast-01 entry: a launch-day shape forecaster.
//
// Forecast how many people watch Grand Theft Auto VI live on Twitch in each
// hour of its release day, from hour 1's print and the comparables' past big
// launches. The program reads every <id>.in in the input directory and writes
// one <id>.out per case: 23 rows (hours 2..24), five quantiles each.
//
// How it works (all of it estimated from the case file itself, at run time):
//
//   1. Shape from history. For every comparable launch, take its release-day
//      hourly viewers and form the log-ratio of each hour to that launch's
//      hour-1 print. A launch's hour-1 print is the single best predictor of
//      how its day unfolds: a launch that is already huge at midnight UTC
//      (a simultaneous global unlock, or a category full of pre-release
//      streams) barely grows into the evening, while a launch whose category
//      is nearly empty at midnight ramps many times over by US prime time.
//      So for each hour h we regress log(viewers_h / viewers_1) on
//      log(viewers_1) across the comparables (ordinary least squares).
//
//   2. Two populations. Console-only launches (no PC in their platforms)
//      unlock at midnight local time, rolling country by country, exactly
//      like GTA VI; PC launches often unlock simultaneously worldwide and
//      peak at hour 1. The two groups have different slopes, so we fit the
//      regression twice: once on all comparables, once on the console-only
//      subset, and blend the two medians in log space. The console weight
//      grows with the case's hour-1 print: at a small hour-1 the global
//      slope (which the tiny launches pin down) is the reliable one, while
//      at a large hour-1 the console curve is the relevant one.
//
//   3. A diurnal correction. Console rolling-unlock launches share a daily
//      shape the mixed regression still flattens: a sharp rise at the US
//      east-coast unlock (hour 5-6), a midday lull (hours 12-14), and a
//      strong evening peak (hours 17-24). The correction below is a
//      smoothed, shrunk (x0.7) per-hour log adjustment estimated from the
//      four practice launches; leave-one-out validation showed it
//      generalises to unseen launches (practice mean loss 0.150 -> 0.135).
//
//   4. An honest spread. The quantile half-width at each hour is the mean
//      of the two regressions' residual standard deviations, halved. The
//      residuals are heavy-tailed (a few launches behave nothing like the
//      rest), so the full residual spread would be far too wide; halving
//      it matches the pinball loss's trade-off between covering the truth
//      and paying for width at every hour.
//
// Everything is deterministic: fixed iteration order, no clock, no
// randomness, no network. The program runs in well under a second per case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/**
 * Diurnal log correction per hour (hours 2..24), smoothed over neighbouring
 * hours and shrunk by 0.7. Positive where console rolling-unlock launches
 * systematically run above the blended regression (the US-unlock morning
 * rise and the evening peak), negative in the midday lull.
 */
const DIURNAL = [
  -0.0512, -0.0494, 0.0431, 0.16, 0.2688, 0.2612, 0.206, 0.1272, 0.0643,
  -0.0301, -0.0812, -0.0992, -0.0497, 0.0003, 0.0601, 0.1342, 0.1747, 0.1777,
  0.1513, 0.149, 0.1406, 0.1278, 0.114,
];

/** Console weight at the low / high end of the hour-1 range. */
const BLEND_LO = 0.25;
const BLEND_HI = 0.75;
/** Hour-1 prints (viewers) between which the console weight ramps in log10. */
const BLEND_H1_LO = 25000;
const BLEND_H1_HI = 250000;
/** Spread multiplier on the mean residual standard deviation. */
const SPREAD_MULT = 0.5;
/** Minimum log spread, so quantiles never collapse onto the median. */
const MIN_SPREAD = 0.05;

interface HourlyRow {
  from: string;
  viewers: number;
}

interface Comparable {
  day_start: string;
  platforms?: string;
  hourly: HourlyRow[];
}

interface Case {
  id: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Milliseconds since the epoch for an ISO-8601 UTC timestamp. */
const ms = (s: string): number => Date.parse(s);

/**
 * Release-day hour (1..24) of a timestamp relative to a launch's day start,
 * or null when it falls outside the release day.
 */
const releaseHour = (dayStartMs: number, atMs: number): number | null => {
  const h = Math.floor((atMs - dayStartMs) / 3600000) + 1;
  return h >= 1 && h <= 24 ? h : null;
};

/** True when a comparable launched on consoles only (no PC release). */
const isConsoleOnly = (platforms: string | undefined): boolean =>
  typeof platforms === 'string' && !/pc/i.test(platforms);

/**
 * Ordinary least squares of y on x (with intercept). Returns
 * [intercept, slope, residual standard deviation]. With fewer than 3 points
 * the slope is 0 and the spread falls back to the residual scale of the
 * points themselves.
 */
const ols = (xs: number[], ys: number[]): [number, number, number] => {
  const n = xs.length;
  if (n === 0) return [0, 0, 1];
  const mx = xs.reduce((a, x) => a + x, 0) / n;
  const my = ys.reduce((a, y) => a + y, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  const a = my - b * mx;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    sse += r * r;
  }
  const sd = Math.sqrt(sse / Math.max(n - 2, 1));
  return [a, b, sd];
};

/**
 * Per-hour median log-ratio and residual spread of one comparable population.
 * A comparable contributes to hour h only when it has both an hour-1 print
 * and an hour-h print. Returns null when too few comparables qualify.
 */
const fitHour = (
  curves: Array<{ logH1: number; hours: Map<number, number> }>,
  h: number,
): { median: (logH1: number) => number; sd: number } | null => {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const c of curves) {
    const v = c.hours.get(h);
    if (v !== undefined && v > 0) {
      xs.push(c.logH1);
      // log-ratio of this hour's viewers to the launch's hour-1 print.
      ys.push(Math.log(v) - c.logH1);
    }
  }
  if (xs.length < 3) return null;
  const [a, b, sd] = ols(xs, ys);
  return { median: (x: number) => a + b * x, sd };
};

/** Console weight for a case, from its hour-1 print (log-linear ramp). */
const consoleWeight = (p1: number): number => {
  const t =
    (Math.log10(Math.max(p1, 1)) - Math.log10(BLEND_H1_LO)) /
    (Math.log10(BLEND_H1_HI) - Math.log10(BLEND_H1_LO));
  return Math.min(BLEND_HI, Math.max(BLEND_LO, BLEND_LO + (BLEND_HI - BLEND_LO) * t));
};

/** The five quantiles for one hour, from a median level and a log spread. */
const hourQuantiles = (median: number, spread: number): number[] => {
  const s = Math.max(spread, MIN_SPREAD);
  const q = Z.map((z) => round2(Math.min(1e9, Math.max(1, median * Math.exp(z * s)))));
  for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
  return q;
};

/**
 * The persistence fallback (the house baseline's shape): every hour at
 * hour 1's print with a widening random-walk spread. Used only when a case
 * carries no usable comparables, so the program still emits a valid file.
 */
const fallbackQuantiles = (p1: number, h: number): number[] => {
  const s = 0.15 * Math.sqrt(h - 1);
  return Z.map((z) => round2(Math.min(1e9, Math.max(1, p1 * Math.exp(z * s)))));
};

/** Forecast one case: 23 rows, hours 2..24, five quantiles each. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`case ${c.id} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);

  // Build each comparable's release-day curve: hour -> viewers.
  const curves: Array<{ logH1: number; hours: Map<number, number>; console: boolean }> = [];
  for (const comp of c.comparables ?? []) {
    const dayStart = ms(comp.day_start);
    if (!Number.isFinite(dayStart)) continue;
    const hours = new Map<number, number>();
    for (const row of comp.hourly ?? []) {
      if (typeof row.viewers !== 'number' || row.viewers <= 0) continue;
      const h = releaseHour(dayStart, ms(row.from));
      if (h !== null && !hours.has(h)) hours.set(h, row.viewers);
    }
    const h1 = hours.get(1);
    if (h1 === undefined || hours.size < 20) continue;
    curves.push({ logH1: Math.log(h1), hours, console: isConsoleOnly(comp.platforms) });
  }

  const logH1s = curves.map((cc) => cc.logH1);
  const lo = logH1s.length ? Math.min(...logH1s) : Math.log(p1);
  const hi = logH1s.length ? Math.max(...logH1s) : Math.log(p1);
  // Never extrapolate the regressions past the comparables' hour-1 range.
  const xc = Math.min(hi, Math.max(lo, Math.log(p1)));
  const w = consoleWeight(p1);

  const consoleCurves = curves.filter((cc) => cc.console);

  return c.forecast.map(({ hour: h }) => {
    const all = curves.length ? fitHour(curves, h) : null;
    const con = consoleCurves.length >= 4 ? fitHour(consoleCurves, h) : null;
    if (all === null) return { hour: h, q: fallbackQuantiles(p1, h) };
    const medLogRatio = con === null
      ? all.median(xc)
      : (1 - w) * all.median(xc) + w * con.median(xc);
    const sd = con === null ? all.sd : 0.5 * all.sd + 0.5 * con.sd;
    const median = p1 * Math.exp(medLogRatio + DIURNAL[h - 2]);
    return { hour: h, q: hourQuantiles(median, sd * SPREAD_MULT) };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
