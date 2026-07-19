import * as RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';
import fs from 'fs';
import { resolve, join } from 'path';
import zlib from 'zlib';

let rapierReady = false;

// ── XGBoost Model (1st place / race winner only) — aligned with training v4 ──
// Training notebook: VelocityBet marble-race XGBoost multi:softprob on 30-tick
// snapshots with 247 features. See model_meta.json next to the booster file.
let xgboostWinnerModel = null;
let xgboostModelMeta = null;
/** Averaged winner path: { points:[{x,y,z,progress,arcLength}], totalArcLength }. */
let trackCenterline = null;

/** Feature / CSV marble order from training (Red == live Black marble). */
const MODEL_FEATURE_MARBLES = [
  'Yellow', 'White', 'Cyan', 'Green', 'Red', 'Blue', 'Orange', 'Purple',
];
/** Class index order = sorted(winner unique) from training label_map. */
const MODEL_CLASS_ORDER = [
  'Blue', 'Cyan', 'Green', 'Orange', 'Purple', 'Red', 'White', 'Yellow',
];
/** Live stream / betting marbleIndex order (frontend RACE_MARBLE_COLORS). */
const LIVE_MARBLE_ORDER = [
  'Green', 'Blue', 'Yellow', 'White', 'Cyan', 'Purple', 'Orange', 'Black',
];
const LIVE_TO_MODEL_NAME = { Black: 'Red' };
const PHYSICS_HZ_PRED = 120;
const RACE_DURATION_SEC_PRED = 45.0;
const PRED_SAMPLE_INTERVAL = 30; // matches collect_data SAMPLE_INTERVAL
const PRED_HISTORY_LEN = 16;     // need ≥10 for roll10 and ≥4 for delta3

function loadXGBoostModels() {
  const candidates = [
    join(process.cwd(), 'xgboost_model_winner.json'),
    join(process.cwd(), 'xgboost_model_winner_v4.json'),
    join(process.cwd(), 'xgboost_model_winner_1.json'),
  ];
  const modelPath = candidates.find((p) => fs.existsSync(p));
  if (!modelPath) {
    console.warn('[Engine] XGBoost model not found. Tried:', candidates.map((p) => p.split(/[/\\]/).pop()).join(', '));
    return;
  }
  xgboostWinnerModel = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
  const nFeat = Number(xgboostWinnerModel?.learner?.learner_model_param?.num_feature || 0);
  const featNames = xgboostWinnerModel?.learner?.feature_names;
  console.log(`[Engine] Loaded XGBoost model: ${modelPath.split(/[/\\]/).pop()} (num_feature=${nFeat || featNames?.length || '?'})`);

  const metaPath = join(process.cwd(), 'model_meta.json');
  if (fs.existsSync(metaPath)) {
    try {
      xgboostModelMeta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
      console.log(`[Engine] Loaded model_meta.json (${xgboostModelMeta.n_features || '?'} features)`);
    } catch (err) {
      console.warn('[Engine] Failed to parse model_meta.json', err);
    }
  }

  if (Array.isArray(featNames) && featNames.length && featNames.length !== 247) {
    console.warn(`[Engine] Expected 247 v4 features, model has ${featNames.length}`);
  }
}

function loadTrackCenterline() {
  const candidates = [
    join(process.cwd(), 'track-centerline.json'),
    resolve('track-centerline.json'),
  ];
  const path = candidates.find((p) => fs.existsSync(p));
  if (!path) {
    console.warn('[Engine] track-centerline.json not found — progress evidence falls back to −Y');
    trackCenterline = null;
    return;
  }
  try {
    const raw = JSON.parse(fs.readFileSync(path, 'utf-8'));
    const points = Array.isArray(raw.points) ? raw.points : [];
    if (points.length < 2) {
      console.warn('[Engine] track-centerline.json has too few points');
      trackCenterline = null;
      return;
    }
    trackCenterline = {
      points,
      totalArcLength: Number(raw.totalArcLength) || points[points.length - 1]?.arcLength || 1,
    };
    console.log(
      `[Engine] Loaded centerline: ${path.split(/[/\\]/).pop()} (${points.length} pts, arc=${trackCenterline.totalArcLength.toFixed(1)})`,
    );
  } catch (err) {
    console.warn('[Engine] Failed to load track-centerline.json', err);
    trackCenterline = null;
  }
}

/**
 * Project a world position onto the centerline polyline.
 * hintProgress (previous estimate) windows the search so hairpins/loops
 * don't snap a marble onto a distant overlapping segment.
 */
function projectOntoCenterline(x, y, z, hintProgress = null) {
  const pts = trackCenterline?.points;
  if (!pts || pts.length < 2) {
    const approx = Math.max(0, Math.min(1, (1200 - y) / 1200));
    return { progress: approx, arcLength: approx * 6000, dist: 0 };
  }

  let iStart = 0;
  let iEnd = pts.length - 2;
  if (hintProgress != null && Number.isFinite(hintProgress)) {
    const lo = Math.max(0, hintProgress - 0.04);
    const hi = Math.min(1, hintProgress + 0.20);
    while (iStart < iEnd && pts[iStart + 1].progress < lo) iStart++;
    while (iEnd > iStart && pts[iEnd].progress > hi) iEnd--;
  }

  let bestD2 = Infinity;
  let bestArc = 0;
  let bestProg = hintProgress ?? 0;

  for (let i = iStart; i <= iEnd; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const abz = b.z - a.z;
    const apx = x - a.x;
    const apy = y - a.y;
    const apz = z - a.z;
    const abLen2 = abx * abx + aby * aby + abz * abz;
    let t = abLen2 > 1e-12 ? (apx * abx + apy * aby + apz * abz) / abLen2 : 0;
    t = Math.max(0, Math.min(1, t));
    const cx = a.x + t * abx;
    const cy = a.y + t * aby;
    const cz = a.z + t * abz;
    const dx = x - cx;
    const dy = y - cy;
    const dz = z - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < bestD2) {
      bestD2 = d2;
      bestArc = a.arcLength + t * (b.arcLength - a.arcLength);
      bestProg = a.progress + t * (b.progress - a.progress);
    }
  }

  let progress = Math.max(0, Math.min(1, bestProg));
  // Mostly monotonic — allow tiny noise, reject big regressions from bad snaps.
  if (hintProgress != null && Number.isFinite(hintProgress)) {
    if (progress < hintProgress - 0.03) progress = hintProgress;
    else if (progress < hintProgress) progress = hintProgress * 0.85 + progress * 0.15;
  }

  return {
    progress,
    arcLength: bestArc,
    dist: Math.sqrt(bestD2),
  };
}

function parseBaseScores(model, numClasses) {
  const raw = model?.learner?.learner_model_param?.base_score;
  const logits = new Array(numClasses).fill(0);
  if (raw == null) return logits;
  try {
    const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (Array.isArray(parsed)) {
      for (let i = 0; i < numClasses; i++) logits[i] = Number(parsed[i]) || 0;
      return logits;
    }
    const single = Number(parsed);
    if (Number.isFinite(single)) return logits.map(() => single);
  } catch {
    const single = Number(raw);
    if (Number.isFinite(single)) return logits.map(() => single);
  }
  return logits;
}

function evaluateXGBoost(features, model) {
  if (!model || !model.learner) return new Array(MODEL_CLASS_ORDER.length).fill(1 / MODEL_CLASS_ORDER.length);
  const booster = model.learner.gradient_booster.model;
  const trees = booster.trees;
  const treeInfo = booster.tree_info;
  const numClasses = Number(model.learner.learner_model_param?.num_class) || MODEL_CLASS_ORDER.length;
  // v4 booster has boost_from_average + per-class base_score — must seed logits.
  const logits = parseBaseScores(model, numClasses);

  for (let i = 0; i < trees.length; i++) {
    const classId = treeInfo[i];
    const tree = trees[i];
    let nodeIdx = 0;
    while (tree.left_children[nodeIdx] !== -1) {
      const featIdx = tree.split_indices[nodeIdx];
      const val = features[featIdx] ?? 0;
      const cond = tree.split_conditions[nodeIdx];
      if (val < cond) {
        nodeIdx = tree.left_children[nodeIdx];
      } else {
        nodeIdx = tree.right_children[nodeIdx];
      }
    }
    logits[classId] += tree.base_weights[nodeIdx];
  }

  let maxLogit = -Infinity;
  for (let i = 0; i < numClasses; i++) if (logits[i] > maxLogit) maxLogit = logits[i];
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sumExp);
}

function emptyState() {
  return { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
}

function liveIndexForModelMarble(modelName) {
  const liveName = modelName === 'Red' ? 'Black' : modelName;
  const idx = LIVE_MARBLE_ORDER.indexOf(liveName);
  return idx >= 0 ? idx : -1;
}

function stateAt(history, liveIdx, histOffsetFromEnd = 0) {
  const snap = history[history.length - 1 - histOffsetFromEnd];
  if (!snap || liveIdx < 0) return emptyState();
  return snap[liveIdx] || emptyState();
}

function speedOf(m) {
  return Math.sqrt(m.vx * m.vx + m.vy * m.vy + m.vz * m.vz);
}

/** Pandas-like rank; ties get average ranks. higherBetter=false → lower values rank 1. */
function rankAxis(values, higherBetter) {
  const n = values.length;
  const order = values
    .map((v, i) => ({ v, i }))
    .sort((a, b) => (higherBetter ? b.v - a.v : a.v - b.v));
  const ranks = new Array(n).fill(0);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && order[j + 1].v === order[i].v) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[order[k].i] = avgRank;
    i = j + 1;
  }
  return ranks;
}

/**
 * Build the exact 247-feature vector used by training v4, in model feature_names order.
 * history = array of snapshot frames in LIVE marble index order.
 */
function buildPredictorFeatures(snapshotHistory, stepCount, raceStartTick = 0) {
  if (!snapshotHistory.length) return null;

  const H = snapshotHistory.length;
  const current = snapshotHistory[H - 1];
  const prev = H > 1 ? snapshotHistory[H - 2] : current;
  const snap3 = H > 3 ? snapshotHistory[H - 4] : current; // diff(3) → 3 steps back

  // Per-model-marble live indices + states
  const liveIdxs = MODEL_FEATURE_MARBLES.map(liveIndexForModelMarble);
  const now = liveIdxs.map((li) => (li >= 0 ? current[li] || emptyState() : emptyState()));
  const was = liveIdxs.map((li) => (li >= 0 ? (prev[li] || emptyState()) : emptyState()));
  const ago3 = liveIdxs.map((li) => (li >= 0 ? (snap3[li] || emptyState()) : emptyState()));
  const speeds = now.map(speedOf);

  const features = [];

  // 1. Raw x,y,z,vx,vy,vz (8×6 = 48)
  for (let i = 0; i < 8; i++) {
    const m = now[i];
    features.push(m.x, m.y, m.z, m.vx, m.vy, m.vz);
  }

  // 2. Per-marble kinematics (8×11 = 88): speed, dv*, speed_roll5, *_roll10, d*_3
  for (let i = 0; i < 8; i++) {
    const mNow = now[i];
    const mPrev = was[i];
    const m3 = ago3[i];
    const li = liveIdxs[i];

    features.push(speeds[i]);
    features.push(mNow.vx - mPrev.vx, mNow.vy - mPrev.vy, mNow.vz - mPrev.vz);

    let speedSum5 = 0;
    const count5 = Math.min(5, H);
    for (let h = 0; h < count5; h++) {
      speedSum5 += speedOf(stateAt(snapshotHistory, li, h));
    }
    features.push(speedSum5 / count5);

    let xSum10 = 0;
    let ySum10 = 0;
    let zSum10 = 0;
    const count10 = Math.min(10, H);
    for (let h = 0; h < count10; h++) {
      const m = stateAt(snapshotHistory, li, h);
      xSum10 += m.x;
      ySum10 += m.y;
      zSum10 += m.z;
    }
    features.push(xSum10 / count10, ySum10 / count10, zSum10 / count10);

    features.push(mNow.x - m3.x, mNow.y - m3.y, mNow.z - m3.z);
  }

  // 3. Ranks & gaps (8×7 = 56) — need previous z/y ranks for delta3
  const yRanks = rankAxis(now.map((m) => m.y), false); // lower y → rank 1
  const zRanks = rankAxis(now.map((m) => m.z), true);  // higher z → rank 1
  const speedRanks = rankAxis(speeds, true);
  const leaderY = Math.min(...now.map((m) => m.y));
  const leaderZ = Math.max(...now.map((m) => m.z));

  const ago3States = liveIdxs.map((li) => stateAt(snapshotHistory, li, Math.min(3, H - 1)));
  const yRanks3 = rankAxis(ago3States.map((m) => m.y), false);
  const zRanks3 = rankAxis(ago3States.map((m) => m.z), true);

  for (let i = 0; i < 8; i++) {
    features.push(yRanks[i]);
    features.push(now[i].y - leaderY);
    features.push(zRanks[i]);
    features.push(leaderZ - now[i].z);
    features.push(speedRanks[i]);
    features.push(zRanks[i] - zRanks3[i]);
    features.push(yRanks[i] - yRanks3[i]);
  }

  // 4. Pairwise 3D distances (28)
  for (let i = 0; i < 8; i++) {
    for (let j = i + 1; j < 8; j++) {
      const a = now[i];
      const b = now[j];
      features.push(Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2));
    }
  }

  // 5. Race progress (3)
  const raceProgressEngine = Math.min(
    1.0,
    (stepCount * (1 / PHYSICS_HZ_PRED)) / RACE_DURATION_SEC_PRED,
  );
  // Live races don't know finish tick — approximate within-race progress with engine formula.
  const raceProgress = raceProgressEngine;
  const ticksSinceStart = Math.max(0, stepCount - raceStartTick);
  features.push(raceProgressEngine, raceProgress, ticksSinceStart);

  // 6. Progress interactions (8×3 = 24)
  const early = 1.0 - raceProgress;
  for (let i = 0; i < 8; i++) {
    features.push(zRanks[i] * raceProgress);
    features.push((leaderZ - now[i].z) * early);
    features.push(speeds[i] * early);
  }

  return features;
}

/** Remap model class-order softmax → live marbleIndex probabilities. */
function probsToLiveOrder(classProbs) {
  const live = new Array(LIVE_MARBLE_ORDER.length).fill(0);
  for (let liveIdx = 0; liveIdx < LIVE_MARBLE_ORDER.length; liveIdx++) {
    const liveName = LIVE_MARBLE_ORDER[liveIdx];
    const modelName = LIVE_TO_MODEL_NAME[liveName] || liveName;
    const classIdx = MODEL_CLASS_ORDER.indexOf(modelName);
    live[liveIdx] = classIdx >= 0 ? classProbs[classIdx] ?? 0 : 0;
  }
  const sum = live.reduce((a, b) => a + b, 0);
  if (sum > 0 && Math.abs(sum - 1) > 1e-6) {
    for (let i = 0; i < live.length; i++) live[i] /= sum;
  }
  return live;
}

function normalizeProbVector(probs) {
  const n = probs.length || 1;
  const out = probs.map((p) => Math.max(1e-12, Number(p) || 0));
  const sum = out.reduce((a, b) => a + b, 0);
  if (sum <= 0) return out.map(() => 1 / n);
  return out.map((p) => p / sum);
}

function softmaxScores(scores, beta = 1) {
  const n = scores.length;
  let maxS = -Infinity;
  for (let i = 0; i < n; i++) if (scores[i] > maxS) maxS = scores[i];
  const exps = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(beta * (scores[i] - maxS));
    exps[i] = e;
    sum += e;
  }
  if (sum <= 0) return exps.map(() => 1 / n);
  return exps.map((e) => e / sum);
}

/**
 * Simple live odds:
 *   P = 0.70 × P_model + 0.30 × P_centerline
 * Centerline = softmax over arc-length progress along the track.
 * Cash-out profit is capped on the API (min(stake, cashOut)) — keep this blend simple.
 */
function blendModelWithCenterline({
  marbleBodies,
  eliminated,
  finished,
  finishOrder,
  modelProbs,
  progressHints,
}) {
  const n = marbleBodies.length;
  const first = finishOrder.length ? finishOrder[0] : -1;
  if (first >= 0) {
    const q = new Array(n).fill(1e-12);
    q[first] = 1;
    return normalizeProbVector(q);
  }

  const progress = new Array(n).fill(0);
  const scores = new Array(n);

  for (let i = 0; i < n; i++) {
    if (eliminated[i] || finished[i] || !marbleBodies[i]) {
      progress[i] = -1;
      scores[i] = -1e6;
      if (progressHints) progressHints[i] = -1;
      continue;
    }
    const p = marbleBodies[i].translation();
    const hint = progressHints?.[i] >= 0 ? progressHints[i] : null;
    const proj = projectOntoCenterline(p.x, p.y, p.z, hint);
    progress[i] = proj.progress;
    if (progressHints) progressHints[i] = proj.progress;
    // Softmax over progress — ~0.15 lead ≈ e^1 relative weight.
    scores[i] = progress[i] / 0.15;
  }

  const pCenter = softmaxScores(scores, 1);
  const pModel = normalizeProbVector(modelProbs || new Array(n).fill(1 / n));
  const blended = pModel.map((pm, i) => 0.7 * pm + 0.3 * pCenter[i]);
  return normalizeProbVector(blended);
}

function runPredictions(features) {
  if (!xgboostWinnerModel || !features) return null;

  const expected =
    Number(xgboostWinnerModel?.learner?.learner_model_param?.num_feature) ||
    xgboostWinnerModel?.learner?.feature_names?.length ||
    247;
  if (features.length !== expected) {
    console.warn(
      `[Engine] Feature length mismatch: got ${features.length}, model expects ${expected}`,
    );
  }

  const classProbs = evaluateXGBoost(features, xgboostWinnerModel);
  const probs = probsToLiveOrder(classProbs);
  const maxProb = Math.max(...probs);
  const winnerId = probs.indexOf(maxProb);

  return [{
    probabilities: probs,
    confidence: maxProb,
    winner_id: winnerId,
  }];
}

// ── In-memory store ──
let storedSnapshot = null;
let storedMetadata = null;

// Pre-restored world — avoids ~25s restoreSnapshot on every race request
let warmedWorld = null;
let warmupPromise = null;

export async function storeSnapshot(snapshotBuffer, metadata) {
  // If the incoming buffer is gzipped, decompress it for the in-memory store
  if (snapshotBuffer[0] === 0x1f && snapshotBuffer[1] === 0x8b) {
    storedSnapshot = zlib.gunzipSync(snapshotBuffer);
    console.log(`[Engine] Decompressed received snapshot from ${snapshotBuffer.length} to ${storedSnapshot.length} bytes`);
    fs.writeFileSync(resolve('track-snapshot.bin.gz'), snapshotBuffer);
  } else {
    storedSnapshot = snapshotBuffer;
    fs.writeFileSync(resolve('track-snapshot.bin'), snapshotBuffer);
  }
  
  storedMetadata = metadata;
  fs.writeFileSync(resolve('track-metadata.json'), JSON.stringify(metadata, null, 2));
  console.log(`[Engine] Snapshot stored in memory: ${storedSnapshot.length} bytes, ${metadata.marbles?.length || 0} marbles`);

  // Invalidate warm world — new snapshot needs a fresh restore
  if (warmedWorld) {
    warmedWorld.free();
    warmedWorld = null;
  }
  warmupPromise = null;
  scheduleRewarm();
}

export async function ensureInitialized() {
  if (!rapierReady) {
    await RAPIER.init({});
    rapierReady = true;
    console.log('[Engine] Rapier initialized');
  }
  if (!xgboostWinnerModel) {
    loadXGBoostModels();
  }
  if (!trackCenterline) {
    loadTrackCenterline();
  }
}

// ── Autonomous Loading ──
export async function autoInitialize() {
  if (storedSnapshot && storedMetadata) return;

  await ensureInitialized();
  const snapshotPath = resolve('track-snapshot.bin');
  const snapshotGzPath = resolve('track-snapshot.bin.gz');
  const metaPath = resolve('track-metadata.json');
  
  if (fs.existsSync(snapshotGzPath) && fs.existsSync(metaPath)) {
    const rawBuffer = fs.readFileSync(snapshotGzPath);
    storedSnapshot = zlib.gunzipSync(rawBuffer);
    storedMetadata = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    console.log(`[Engine] Autonomous Init: Loaded and decompressed snapshot (${storedSnapshot.length} bytes) and metadata.`);
  } else if (fs.existsSync(snapshotPath) && fs.existsSync(metaPath)) {
    storedSnapshot = fs.readFileSync(snapshotPath);
    storedMetadata = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    console.log(`[Engine] Autonomous Init: Loaded snapshot (${storedSnapshot.length} bytes) and metadata.`);
  } else {
    throw new Error('Static assets track-snapshot.bin(.gz) or track-metadata.json missing.');
  }
}

// ── Constants (matched to frontend) ──
const OBSTACLE_SINK_DEPTH = 14;
// Transit times are fixed for every pebble — no per-obstacle compromise.
const OBSTACLE_SINK_TIME = 2.0;
const OBSTACLE_RISE_TIME = 2.0;
// Brief dwell for normal pebbles so marbles don't sit stuck on raised pegs.
const OBSTACLE_HOLD_TIME = 0.25;
const OBSTACLE_WAIT_TIME = 0.25;
// Finish-plane / spiral pebbles may linger longer at top/bottom.
const OBSTACLE_HOLD_TIME_LONG = 2.5;
const OBSTACLE_WAIT_TIME_LONG = 4.0;
const PHYSICS_STEP = 1 / 120;
const PHYSICS_HZ = 120;
// Client playback multiplier (engine streams as fast as physics computes — no wall-clock sleep).
const STREAM_SPEED = 2.0;
const KEYFRAME_INTERVAL = 1; // every physics tick streamed; client interpolates + display-gates
const MARBLE_SETTLE_FRAMES = 90;
// Authoritative race timeline (in physics ticks). The frontend derives the
// 3-2-1-GO countdown and the gate opening from these so visuals match physics.
const COUNTDOWN_START_TICK = 120;
const COUNTDOWN_DURATION_TICKS = 420;
const GATE_OPEN_TICK = COUNTDOWN_START_TICK + COUNTDOWN_DURATION_TICKS; // 540
const IMPULSE_TICK = GATE_OPEN_TICK;

function applyWorldDefaults(world) {
  world.gravity = { x: 0, y: -9.81 * 15.2, z: 0 };
  world.timestep = PHYSICS_STEP;
}

/** Load track data and restore the physics world ahead of the first race request. */
export async function startWarmup() {
  if (warmedWorld) return warmedWorld;
  if (warmupPromise) return warmupPromise;

  warmupPromise = (async () => {
    await autoInitialize();
    const t = Date.now();
    console.log(`[Engine] Pre-warming physics world from ${storedSnapshot.length} byte snapshot...`);
    const world = RAPIER.World.restoreSnapshot(storedSnapshot);
    applyWorldDefaults(world);
    warmedWorld = world;
    console.log(`[Engine] Pre-warm complete in ${Date.now() - t}ms`);
    warmupPromise = null;
    return warmedWorld;
  })();

  return warmupPromise;
}

export function getWarmupStatus() {
  return {
    ready: warmedWorld !== null,
    warming: warmupPromise !== null,
    hasSnapshot: !!(storedSnapshot && storedMetadata),
  };
}

function scheduleRewarm() {
  startWarmup().catch((err) => {
    console.error('[Engine] Background re-warm failed:', err.message);
  });
}

// ── Seeded PRNG (exact replica of frontend) ──
function mulberry32(a) {
  return function () {
    var t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const _bladeAxis = new THREE.Vector3(1, 0, 0);
const _bladeRotOffset = new THREE.Quaternion();
const _tempQ = new THREE.Quaternion();
const _tempVec = new THREE.Vector3();

function resolveBodyHandle(handle) {
  if (handle === -1 || handle == null) return -1;
  if (typeof handle === 'string') return parseFloat(handle);
  return handle;
}

/** Grid position slightly behind the gate hinge — matches frontend visual grid. */
function marbleGridPos(m) {
  const sp = m.startPos;
  return { x: sp.x, y: sp.y, z: sp.z - 3.0 };
}

/** Fisher–Yates shuffle of start lanes (colors stay on marble index; only X lanes move). */
function shuffleMarbleStarts(marbles, rng) {
  const starts = marbles.map((m) => marbleGridPos(m));
  for (let i = starts.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = starts[i];
    starts[i] = starts[j];
    starts[j] = tmp;
  }
  return starts;
}

async function yieldFramePace() {
  // No-op: the API pace-gate controls delivery timing.
  // The engine produces frames as fast as possible into the API's buffer.
}

// Yield the event loop periodically to stay responsive to health checks etc.
async function yieldEventLoop() {
  await new Promise((r) => setImmediate(r));
}

export async function* streamRace(seed) {
  const _startTime = Date.now();
  console.log(`[Engine] streamRace initiated at +0ms`);

  await startWarmup();

  const meta = storedMetadata;
  const seedNum = typeof seed === 'number' ? seed : (typeof seed === 'string' ? parseInt(seed, 10) : seed);
  
  const obstacleRandom = mulberry32(seedNum + 12345);
  const marbleRandom = mulberry32(seedNum + 54321);
  const laneRandom = mulberry32(seedNum + 77777);

  for (let i = 0; i < 7; i++) marbleRandom();

  // Per-race lane order (provably tied to race seed). Metadata startPos alone is fixed.
  const shuffledStarts = shuffleMarbleStarts(meta.marbles, laneRandom);
  console.log(
    `[Engine] Start lanes reshuffled:`,
    shuffledStarts.map((p) => p.x.toFixed(1)).join(', '),
  );

  let physicsWorld;
  if (warmedWorld) {
    physicsWorld = warmedWorld;
    warmedWorld = null;
    console.log(`[Engine] Using pre-warmed world at +${Date.now() - _startTime}ms`);
  } else {
    console.log(`[Engine] Restoring Rapier snapshot at +${Date.now() - _startTime}ms...`);
    const _restoreStart = Date.now();
    physicsWorld = RAPIER.World.restoreSnapshot(storedSnapshot);
    applyWorldDefaults(physicsWorld);
    console.log(`[Engine] Restored Rapier snapshot in ${Date.now() - _restoreStart}ms`);
  }

  function getBody(handle) {
    const h = resolveBodyHandle(handle);
    return h !== -1 ? physicsWorld.getRigidBody(h) : null;
  }

  console.log(`[Engine] Mapping bodies at +${Date.now() - _startTime}ms...`);
  const marbleBodies = meta.marbles.map(m => getBody(m.handle));
  const obstacleBodies = meta.obstacles.map(o => getBody(o.handle));
  const bladeBodies = meta.blades.map(b => getBody(b.handle));
  const gateBody = getBody(meta.handles.gate);

  if (!meta.lastRunBB?.min || !meta.lastRunBB?.max) {
    console.warn('[Engine] track-metadata.json has no lastRunBB — spiral buy-lock disabled until regions are extracted from the GLB');
  } else {
    console.log(
      `[Engine] lastRunBB lock volume y=${meta.lastRunBB.min.y.toFixed(1)}..${meta.lastRunBB.max.y.toFixed(1)}`,
    );
  }

  function setRestitution(body, value) {
    if (!body) return;
    for (let j = 0; j < body.numColliders(); j++) {
      body.collider(j).setRestitution(value);
    }
  }
  marbleBodies.forEach(b => setRestitution(b, 0.12));
  obstacleBodies.forEach(b => setRestitution(b, 0.1));
  bladeBodies.forEach(b => setRestitution(b, 0.12));
  setRestitution(gateBody, 0.05);

  // Place each marble on its reshuffled start lane before the first streamed frame.
  for (let i = 0; i < marbleBodies.length; i++) {
    const body = marbleBodies[i];
    const sp = shuffledStarts[i];
    if (!body || !sp) continue;
    body.setTranslation(sp, true);
    body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  // Lower linear damping so marbles keep enough momentum to climb inclines
  marbleBodies.forEach(b => { if (b) b.setLinearDamping(0.01); });

  function obstacleInBB(oMeta, bb, { yPad = 0 } = {}) {
    if (!bb?.min || !bb?.max || !oMeta) return false;
    return (
      oMeta.wx >= bb.min.x && oMeta.wx <= bb.max.x &&
      oMeta.bodyWy >= bb.min.y - yPad && oMeta.bodyWy <= bb.max.y + yPad &&
      oMeta.wz >= bb.min.z && oMeta.wz <= bb.max.z
    );
  }

  function isLongStayObstacle(oMeta) {
    // Finish plane is a thin slab — pad Y so pegs sitting on it still match.
    // lastRunBB covers the final spiral section.
    return (
      obstacleInBB(oMeta, meta.finishPlaneBB, { yPad: 25 }) ||
      obstacleInBB(oMeta, meta.lastRunBB, { yPad: 10 })
    );
  }

  function obstacleWaitTime(oMeta) {
    return isLongStayObstacle(oMeta) ? OBSTACLE_WAIT_TIME_LONG : OBSTACLE_WAIT_TIME;
  }

  function obstacleHoldTime(oMeta) {
    return isLongStayObstacle(oMeta) ? OBSTACLE_HOLD_TIME_LONG : OBSTACLE_HOLD_TIME;
  }

  // Desync start phases only — sink/rise always use the full 2s once entered.
  const obsState = meta.obstacles.map((oMeta) => {
    const rnd = obstacleRandom();
    const waitT = obstacleWaitTime(oMeta);
    const holdT = obstacleHoldTime(oMeta);
    let state = 'waiting';
    let timer = 0;
    if (rnd < 0.25) { state = 'waiting'; timer = obstacleRandom() * waitT; }
    else if (rnd < 0.5) { state = 'sinking'; timer = obstacleRandom() * OBSTACLE_SINK_TIME; }
    else if (rnd < 0.75) { state = 'sunk'; timer = obstacleRandom() * holdT; }
    else { state = 'rising'; timer = obstacleRandom() * OBSTACLE_RISE_TIME; }
    return { timer, state };
  });

  const trackForwardDir = new THREE.Vector3(meta.trackForwardDir.x, meta.trackForwardDir.y, meta.trackForwardDir.z);

  let physicsTick = 0;
  let physicsTimeTotal = 0;
  let gateIsOpen = false;
  let countdownStarted = false;
  let gateOpenTick = -1;
  let impulseTick = -1;
  const maxPrecomputeTicks = 120 * 300;
  const finishedMarbles = [];
  const eliminated = meta.marbles.map(() => false);
  const finished = meta.marbles.map(() => false);
  /** Finish order (first index = race winner for markets). */
  const finishOrder = [];
  const fallTimer = meta.marbles.map(() => 0);
  const stuckTimer = meta.marbles.map(() => 0);
  const kickCooldown = meta.marbles.map(() => 0);

  // ── Prediction state ──
  const predictionHistory = [];
  const centerlineProgressHints = meta.marbles.map(() => 0);
  let lastPredictions = null;
  // Sticky: once any marble enters the mesh-derived last-run volume, lock buys.
  let bettingLocked = false;

  function pointInRegionBB(pos, bb, { yMaxPad = 0 } = {}) {
    if (!bb?.min || !bb?.max) return false;
    return (
      pos.x >= bb.min.x && pos.x <= bb.max.x &&
      pos.y >= bb.min.y && pos.y <= bb.max.y + yMaxPad &&
      pos.z >= bb.min.z && pos.z <= bb.max.z
    );
  }

  function anyMarbleInLastSpiral() {
    if (!gateIsOpen) return false;
    const bb = meta.lastRunBB;
    if (!bb?.min || !bb?.max) return false;
    for (let i = 0; i < marbleBodies.length; i++) {
      const body = marbleBodies[i];
      if (!body || eliminated[i] || finished[i]) continue;
      // Same named-mesh AABB the track export derives (last + run). Slight
      // roof pad matches camera spiral framing when approaching from above.
      if (pointInRegionBB(body.translation(), bb, { yMaxPad: 10 })) return true;
    }
    return false;
  }

  console.log(`[Engine] Setup complete. Yielding 'start' chunk at +${Date.now() - _startTime}ms`);
  yield {
    type: 'start',
    seed: seedNum,
    physicsStep: PHYSICS_STEP,
    physicsHz: PHYSICS_HZ,
    keyframeInterval: KEYFRAME_INTERVAL,
    countdownStartTick: COUNTDOWN_START_TICK,
    gateOpenTick: GATE_OPEN_TICK,
    impulseTick: IMPULSE_TICK,
    streamSpeed: STREAM_SPEED,
    marbleStarts: shuffledStarts,
  };

  const buildFrame = (tick, gateOpen) => ({
    tick,
    marbles: meta.marbles.map((m, i) => {
      const body = marbleBodies[i];
      if (!body) return null;
      const p = body.translation();
      const r = body.rotation();
      const v = body.linvel();
      return [p.x, p.y, p.z, r.x, r.y, r.z, r.w, eliminated[i] ? 1 : 0, finished[i] ? 1 : 0, v.x, v.y, v.z];
    }),
    obstacles: meta.obstacles.map((obs, i) => { const body = obstacleBodies[i]; return body ? body.translation().y : 0; }),
    blades: meta.blades.map((b, i) => { const body = bladeBodies[i]; if (!body) return [0, 0, 0, 1]; const r = body.rotation(); return [r.x, r.y, r.z, r.w]; }),
    gateIsOpen: gateOpen,
    bettingLocked,
  });

  yield { type: 'frame', frame: buildFrame(0, false) };

  // ── Main simulation loop ──
  const targetFinishedCount = 4;
  
  // Finish plane Y-bounds detected from frontend debug logs
  const FINISH_PLANE_MIN_Y = 15; // Replace with actual value from console
  const FINISH_PLANE_MAX_Y = 25; // Replace with actual value from console

  while (true) {
    if (physicsTick >= maxPrecomputeTicks) break;
    const finishedCount = finished.filter(f => f).length;
    if (finishedCount >= targetFinishedCount) break;
    
    physicsTick++;

    physicsTimeTotal += PHYSICS_STEP;

    if (!countdownStarted && !gateIsOpen && marbleBodies.length > 0) {
      if (physicsTick === COUNTDOWN_START_TICK) {
        countdownStarted = true;
        gateOpenTick = GATE_OPEN_TICK;
        impulseTick = IMPULSE_TICK;
      }
    }

    if (physicsTick === gateOpenTick) {
      gateIsOpen = true;
      if (gateBody) {
        gateBody.setNextKinematicTranslation({ x: 0, y: -1000, z: 0 });
      }
      for (let i = 0; i < marbleBodies.length; i++) {
        const body = marbleBodies[i];
        if (!body || eliminated[i]) continue;
        const marbleR = meta.marbles[i]?.radius || 1;
        const forwardImpulse = marbleR * 20.0;
        const jitter = (marbleRandom() - 0.5) * marbleR * 3.0;
        body.applyImpulse(
          {
            x: trackForwardDir.x * forwardImpulse + jitter,
            y: 0,
            z: trackForwardDir.z * forwardImpulse + jitter,
          },
          true,
        );
      }
    }

    if (gateIsOpen) {
      for (let i = 0; i < meta.obstacles.length; i++) {
        const obs = obsState[i];
        const body = obstacleBodies[i];
        const oMeta = meta.obstacles[i];
        if (!body || !oMeta) continue;

        obs.timer -= PHYSICS_STEP;
        let sinkFrac = 0;

        switch (obs.state) {
          case 'waiting':
            if (obs.timer <= 0) { obs.state = 'sinking'; obs.timer = OBSTACLE_SINK_TIME; }
            sinkFrac = 0;
            break;
          case 'sinking':
            sinkFrac = 1 - obs.timer / OBSTACLE_SINK_TIME;
            if (obs.timer <= 0) { obs.state = 'sunk'; obs.timer = obstacleHoldTime(oMeta); sinkFrac = 1; }
            break;
          case 'sunk':
            sinkFrac = 1;
            if (obs.timer <= 0) { obs.state = 'rising'; obs.timer = OBSTACLE_RISE_TIME; }
            break;
          case 'rising':
            sinkFrac = obs.timer / OBSTACLE_RISE_TIME;
            if (obs.timer <= 0) { obs.state = 'waiting'; obs.timer = obstacleWaitTime(oMeta); sinkFrac = 0; }
            break;
        }

        const s = sinkFrac * sinkFrac * (3 - 2 * sinkFrac);
        const sinkAmt = s * OBSTACLE_SINK_DEPTH;
        body.setNextKinematicTranslation({ x: oMeta.wx, y: oMeta.bodyWy - sinkAmt, z: oMeta.wz });
      }

      for (let i = 0; i < meta.blades.length; i++) {
        const blade = meta.blades[i];
        const body = bladeBodies[i];
        if (!body || !blade.baseQuat) continue;
        const bq = blade.baseQuat;
        const randomPhase = blade.mesh ? (blade.mesh.id % 10) * 1.2 : i * 1.5;
        _bladeRotOffset.setFromAxisAngle(_bladeAxis, physicsTimeTotal + randomPhase);
        _tempQ.set(bq.x, bq.y, bq.z, bq.w).multiply(_bladeRotOffset);
        body.setNextKinematicRotation({ x: _tempQ.x, y: _tempQ.y, z: _tempQ.z, w: _tempQ.w });
      }
    }

    physicsWorld.step();

    if (!gateIsOpen) {
      for (let i = 0; i < marbleBodies.length; i++) {
        const body = marbleBodies[i];
        const grid = shuffledStarts[i];
        if (!body || !grid || eliminated[i] || finished[i]) continue;
        body.setTranslation(grid, true);
        body.setLinvel({ x: 0, y: 0, z: 0 }, true);
        body.setAngvel({ x: 0, y: 0, z: 0 }, true);
      }
    }

    if (gateIsOpen && meta.bendBB) {
      for (let i = 0; i < marbleBodies.length; i++) {
        const body = marbleBodies[i];
        if (!body || eliminated[i] || finished[i]) continue;
        const pos = body.translation();
        if (pos.x >= meta.bendBB.min.x && pos.x <= meta.bendBB.max.x &&
            pos.y >= meta.bendBB.min.y && pos.y <= meta.bendBB.max.y &&
            pos.z >= meta.bendBB.min.z && pos.z <= meta.bendBB.max.z) {
          const vel = body.linvel();
          const speed = Math.sqrt(vel.x * vel.x + vel.z * vel.z);
          if (speed > 0.5 && speed < 18) {
            console.log('BEND BOOST TRIGGERED');
            body.setLinvel({ x: vel.x * 2.30, y: vel.y, z: vel.z * 2.30 }, true);
          }
        }
      }
    }

    if (gateIsOpen && meta.glassyExtraBB) {
      for (let i = 0; i < marbleBodies.length; i++) {
        const body = marbleBodies[i];
        if (!body || eliminated[i] || finished[i]) continue;
        const pos = body.translation();
        if (pos.x >= meta.glassyExtraBB.min.x && pos.x <= meta.glassyExtraBB.max.x &&
            pos.y >= meta.glassyExtraBB.min.y && pos.y <= meta.glassyExtraBB.max.y &&
            pos.z >= meta.glassyExtraBB.min.z && pos.z <= meta.glassyExtraBB.max.z) {
          const vel = body.linvel();
          const speed = Math.sqrt(vel.x * vel.x + vel.z * vel.z);
          if (pos.y > 320 && speed < 18.0) {
            if (speed > 0.2) {
              body.setLinvel({ x: (vel.x / speed) * 25.0, y: vel.y, z: (vel.z / speed) * 25.0 }, true);
            } else {
              body.setLinvel({ x: 0.968 * 25.0, y: vel.y, z: 0.252 * 25.0 }, true);
            }
          }
        }
      }
    }

    if (gateIsOpen) {
      for (let i = 0; i < marbleBodies.length; i++) {
        const body = marbleBodies[i];
        if (!body || eliminated[i] || finished[i]) continue;
        
        if (kickCooldown[i] > 0) {
          kickCooldown[i]--;
          continue;
        }
        
        const vel = body.linvel();
        const speed = Math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        const r = meta.marbles[i]?.radius || 1;
        
        if (speed < r * 0.4) {
          stuckTimer[i]++;
          if (stuckTimer[i] >= 90) {
            const kick = r * 9;
            const angle = marbleRandom() * Math.PI * 2;
            body.applyImpulse({
              x: Math.cos(angle) * kick,
              y: kick * 0.4,
              z: Math.sin(angle) * kick,
            }, true);
            stuckTimer[i] = 0;
            kickCooldown[i] = 120;
          }
        } else {
          stuckTimer[i] = 0;
        }
      }
    }

    for (let i = 0; i < marbleBodies.length; i++) {
      const body = marbleBodies[i];
      if (eliminated[i] || finished[i] || !body) continue;
      const p = body.translation();
      
      if (p.x > 550 && p.x < 620 && p.y < 120 && p.y > 40 && physicsTick % 10 === 0) {
        console.log(`[Engine] Marble ${i} at finish zone: x=${p.x.toFixed(2)}, y=${p.y.toFixed(2)}, z=${p.z.toFixed(2)}`);
      }

      let isFinished = false;
      if (meta.finishPlaneBB) {
        // Marbles move along the Z axis towards the finish plane.
        // We restrict p.z >= meta.finishPlaneBB.min.z + 5.0 so it only triggers once they actually cross the start of the checkerboard.
        if (p.x >= meta.finishPlaneBB.min.x - 20.0 && p.x <= meta.finishPlaneBB.max.x + 20.0 &&
            p.y >= meta.finishPlaneBB.min.y - 15.0 && p.y <= meta.finishPlaneBB.max.y + 15.0 &&
            p.z >= meta.finishPlaneBB.min.z + 5.0 && p.z <= meta.finishPlaneBB.max.z - 20.0) {
          isFinished = true;
        }
      } else if (p.y < 30 && p.y > -10) {
        isFinished = true;
      }
      
      if (isFinished) {
        finished[i] = true;
        if (!finishOrder.includes(i)) finishOrder.push(i);
      }
      
      const vel = body.linvel();
      if (vel.y < -110 && Math.abs(vel.x) < 40 && Math.abs(vel.z) < 40) fallTimer[i] += PHYSICS_STEP;
      else fallTimer[i] = 0;
      if (fallTimer[i] > 1.0 || p.y < -50) {
        eliminated[i] = true;
        body.setTranslation({ x: 0, y: -1000, z: 0 }, true);
      }
    }

    if (!bettingLocked && anyMarbleInLastSpiral()) {
      bettingLocked = true;
      console.log(`[Engine] Betting locked — marble entered last spiral at tick ${physicsTick}`);
    }

    const frame = buildFrame(physicsTick, gateIsOpen);

    if (lastPredictions && physicsTick % 60 === 0) {
      frame.predictions = lastPredictions;
    }

    yield { type: 'frame', frame };

    // Yield event loop every 60 frames (~0.5s of sim time) to stay responsive.
    if (physicsTick % 60 === 0) await yieldEventLoop();

    if (gateIsOpen && physicsTick % PRED_SAMPLE_INTERVAL === 0) {
      const snapFrame = meta.marbles.map((m, i) => {
        const body = marbleBodies[i];
        if (!body || eliminated[i] || finished[i]) return { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
        const p = body.translation();
        const v = body.linvel();
        return { x: p.x, y: p.y, z: p.z, vx: v.x, vy: v.y, vz: v.z };
      });
      predictionHistory.push(snapFrame);
      if (predictionHistory.length > PRED_HISTORY_LEN) predictionHistory.shift();

      const raceStartTick = gateOpenTick > 0 ? gateOpenTick : 0;
      const features = buildPredictorFeatures(
        predictionHistory,
        physicsTick,
        raceStartTick,
      );
      const rawPred = runPredictions(features);
      if (rawPred?.[0]?.probabilities) {
        const blended = blendModelWithCenterline({
          marbleBodies,
          eliminated,
          finished,
          finishOrder,
          modelProbs: rawPred[0].probabilities,
          progressHints: centerlineProgressHints,
        });
        const maxProb = Math.max(...blended);
        lastPredictions = [{
          probabilities: blended,
          confidence: maxProb,
          winner_id: blended.indexOf(maxProb),
        }];
      } else {
        lastPredictions = rawPred;
      }
    }
    
    let activeCount = 0;
    for (let i = 0; i < marbleBodies.length; i++) {
      if (!eliminated[i] && !finished[i]) activeCount++;
    }
    if (activeCount === 0) break;
  }

  physicsWorld.free();
  scheduleRewarm();
  yield { type: 'end' };
}
