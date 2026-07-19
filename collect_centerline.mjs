/**
 * Run N headless races via the production engine, record the winner's path
 * from gate-open to finish plane, average those paths into a centerline, and save.
 *
 * Usage (from velocityBet-engine/):
 *   node collect_centerline.mjs
 *   node collect_centerline.mjs --races=10 --points=500
 */
import fs from 'fs';
import { resolve } from 'path';
import { streamRace } from './src/backend/engine.mjs';

const LIVE_MARBLE_NAMES = [
  'Green', 'Blue', 'Yellow', 'White', 'Cyan', 'Purple', 'Orange', 'Black',
];

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [k, v] = arg.replace(/^--/, '').split('=');
    return [k, v ?? 'true'];
  }),
);

const NUM_RACES = Number(args.races) || 10;
const NUM_CENTERLINE_POINTS = Number(args.points) || 500;
const OUTPUT_PATH = resolve(args.out || 'track-centerline.json');
const RAW_OUTPUT_PATH = resolve(args.raw || 'centerline_races_raw.json');

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** Linear interpolation on normalized race progress; null if u outside trajectory span. */
function interpAtProgress(samples, u) {
  if (!samples.length) return null;
  if (u < samples[0].p || u > samples[samples.length - 1].p) return null;
  if (u === samples[0].p) return { x: samples[0].x, y: samples[0].y, z: samples[0].z };

  for (let i = 0; i < samples.length - 1; i++) {
    const a = samples[i];
    const b = samples[i + 1];
    if (u >= a.p && u <= b.p) {
      const denom = b.p - a.p;
      const t = denom > 0 ? (u - a.p) / denom : 0;
      return {
        x: lerp(a.x, b.x, t),
        y: lerp(a.y, b.y, t),
        z: lerp(a.z, b.z, t),
      };
    }
  }
  return null;
}

async function collectRace(seed) {
  const marbleTrajectories = Array.from({ length: 8 }, () => []);
  const wasFinished = Array(8).fill(false);
  let gateOpenTick = null;
  let seedNum = seed;
  let winnerId = -1;
  let winnerFinishTick = null;

  for await (const chunk of streamRace(seed)) {
    if (chunk.type === 'start') {
      gateOpenTick = chunk.gateOpenTick;
      seedNum = chunk.seed;
    } else if (chunk.type === 'frame') {
      const { frame } = chunk;
      if (!frame.gateIsOpen || gateOpenTick == null) continue;

      for (let i = 0; i < frame.marbles.length; i++) {
        const m = frame.marbles[i];
        if (!m) continue;
        const eliminated = m[7] === 1;
        const finished = m[8] === 1;
        const pt = { tick: frame.tick, x: m[0], y: m[1], z: m[2] };

        if (finished && !wasFinished[i]) {
          wasFinished[i] = true;
          if (winnerId === -1) {
            winnerId = i;
            winnerFinishTick = frame.tick;
          }
          marbleTrajectories[i].push(pt);
        } else if (!eliminated && !finished) {
          marbleTrajectories[i].push(pt);
        }
      }
    } else if (chunk.type === 'end') {
      break;
    }
  }

  if (winnerId < 0 || winnerFinishTick == null) {
    throw new Error(`Race seed=${seedNum}: no winner detected`);
  }

  const durationTicks = Math.max(1, winnerFinishTick - gateOpenTick);
  const winnerTrajectory = marbleTrajectories[winnerId].map((pt) => ({
    p: (pt.tick - gateOpenTick) / durationTicks,
    x: pt.x,
    y: pt.y,
    z: pt.z,
  }));

  return {
    seed: seedNum,
    gateOpenTick,
    winnerFinishTick,
    durationTicks,
    winnerId,
    winnerName: LIVE_MARBLE_NAMES[winnerId] ?? `marble_${winnerId}`,
    winnerSampleCount: winnerTrajectory.length,
    winnerTrajectory,
  };
}

/** Resample a single winner path to fixed progress steps. */
function resampleTrajectory(trajectory, numPoints) {
  const points = [];
  for (let k = 0; k <= numPoints; k++) {
    const u = k / numPoints;
    const pt = interpAtProgress(trajectory, u);
    if (!pt) continue;
    points.push({ progress: u, x: pt.x, y: pt.y, z: pt.z });
  }
  return points;
}

function averageCurves(curves) {
  const n = curves[0].length;
  const averaged = [];
  for (let k = 0; k < n; k++) {
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (const curve of curves) {
      sx += curve[k].x;
      sy += curve[k].y;
      sz += curve[k].z;
    }
    averaged.push({
      progress: curves[0][k].progress,
      x: sx / curves.length,
      y: sy / curves.length,
      z: sz / curves.length,
    });
  }
  return averaged;
}

function addArcLength(points) {
  if (!points.length) return [];
  const result = [{ ...points[0], arcLength: 0 }];
  let s = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    const dz = points[i].z - points[i - 1].z;
    s += Math.sqrt(dx * dx + dy * dy + dz * dz);
    result.push({ ...points[i], arcLength: s });
  }
  return result;
}

async function main() {
  console.log(`[Centerline] Running ${NUM_RACES} races → ${NUM_CENTERLINE_POINTS} points (winner paths only)`);
  const t0 = Date.now();
  const seeds = Array.from({ length: NUM_RACES }, (_, i) => 10_000 + i * 7919);
  const races = [];

  for (let i = 0; i < NUM_RACES; i++) {
    console.log(`[Centerline] Race ${i + 1}/${NUM_RACES} (seed=${seeds[i]})`);
    const race = await collectRace(seeds[i]);
    races.push(race);
    console.log(
      `  winner=${race.winnerName} (idx ${race.winnerId}), duration=${race.durationTicks} ticks, samples=${race.winnerSampleCount}`,
    );
  }

  const curves = races.map((r) => resampleTrajectory(r.winnerTrajectory, NUM_CENTERLINE_POINTS));
  const centerline = addArcLength(averageCurves(curves));

  const output = {
    generatedAt: new Date().toISOString(),
    numRaces: NUM_RACES,
    seeds,
    winners: races.map((r) => ({ seed: r.seed, marbleIndex: r.winnerId, name: r.winnerName })),
    numPoints: centerline.length,
    totalArcLength: centerline[centerline.length - 1]?.arcLength ?? 0,
    method:
      'Per race: record only the winner marble from gate-open to finish-plane crossing (progress 0→1). Average those winner paths across races.',
    points: centerline,
  };

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(output, null, 2));
  console.log(
    `[Centerline] Saved ${OUTPUT_PATH} (${centerline.length} points, arc length ${output.totalArcLength.toFixed(1)})`,
  );

  const rawSummary = {
    generatedAt: output.generatedAt,
    races: races.map((r) => ({
      seed: r.seed,
      winnerId: r.winnerId,
      winnerName: r.winnerName,
      gateOpenTick: r.gateOpenTick,
      winnerFinishTick: r.winnerFinishTick,
      durationTicks: r.durationTicks,
      winnerSampleCount: r.winnerSampleCount,
    })),
  };
  fs.writeFileSync(RAW_OUTPUT_PATH, JSON.stringify(rawSummary, null, 2));
  console.log(`[Centerline] Saved ${RAW_OUTPUT_PATH}`);
  console.log(`[Centerline] Done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

main().catch((err) => {
  console.error('[Centerline] Fatal:', err);
  process.exit(1);
});
