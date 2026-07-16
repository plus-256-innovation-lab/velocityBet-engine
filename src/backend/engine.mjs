import * as RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';
import fs from 'fs';
import { resolve, join } from 'path';
import zlib from 'zlib';

let rapierReady = false;

// ── XGBoost Model (1st place / race winner only) ──
let xgboostWinnerModel = null;

function loadXGBoostModels() {
  const modelPath = join(process.cwd(), 'xgboost_model_winner_1.json');
  if (fs.existsSync(modelPath)) {
    xgboostWinnerModel = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
    console.log('[Engine] Loaded XGBoost model: xgboost_model_winner_1.json');
  } else {
    console.warn(`[Engine] XGBoost model not found: ${modelPath}`);
  }
}

function evaluateXGBoost(features, model) {
  if (!model || !model.learner) return [0, 0, 0, 0, 0, 0, 0, 0];
  const booster = model.learner.gradient_booster.model;
  const trees = booster.trees;
  const treeInfo = booster.tree_info;
  const numClasses = 8;
  const logits = new Array(numClasses).fill(0);

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
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExp);
}

function buildPredictorFeatures(snapshotHistory, stepCount) {
  if (snapshotHistory.length < 1) return null;

  const current = snapshotHistory[snapshotHistory.length - 1];
  const prev = snapshotHistory.length > 1 ? snapshotHistory[snapshotHistory.length - 2] : current;
  const features = [];

  // 1. Raw positions/velocities (8 * 6 = 48)
  for (let i = 0; i < 8; i++) {
    const m = current[i] || { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
    features.push(m.x, m.y, m.z, m.vx, m.vy, m.vz);
  }

  // 2. Engineered features (8 * 8 = 64)
  for (let i = 0; i < 8; i++) {
    const mNow = current[i] || { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
    const mPrev = prev[i] || { vx: 0, vy: 0, vz: 0 };
    features.push(mNow.vx - mPrev.vx, mNow.vy - mPrev.vy, mNow.vz - mPrev.vz);
    const speedNow = Math.sqrt(mNow.vx * mNow.vx + mNow.vy * mNow.vy + mNow.vz * mNow.vz);
    features.push(speedNow);
    let speedSum5 = 0;
    const count5 = Math.min(5, snapshotHistory.length);
    for (let h = 0; h < count5; h++) {
      const snap = snapshotHistory[snapshotHistory.length - 1 - h];
      const m = snap[i] || { vx: 0, vy: 0, vz: 0 };
      speedSum5 += Math.sqrt(m.vx * m.vx + m.vy * m.vy + m.vz * m.vz);
    }
    features.push(speedSum5 / count5);
    let xSum10 = 0, ySum10 = 0, zSum10 = 0;
    const count10 = Math.min(10, snapshotHistory.length);
    for (let h = 0; h < count10; h++) {
      const snap = snapshotHistory[snapshotHistory.length - 1 - h];
      const m = snap[i] || { x: 0, y: 0, z: 0 };
      xSum10 += m.x;
      ySum10 += m.y;
      zSum10 += m.z;
    }
    features.push(xSum10 / count10, ySum10 / count10, zSum10 / count10);
  }

  // 3. Pairwise Distances (28)
  for (let i = 0; i < 8; i++) {
    for (let j = i + 1; j < 8; j++) {
      const mi = current[i] || { x: 0, y: 0, z: 0 };
      const mj = current[j] || { x: 0, y: 0, z: 0 };
      features.push(Math.sqrt((mi.x - mj.x) ** 2 + (mi.y - mj.y) ** 2 + (mi.z - mj.z) ** 2));
    }
  }

  // 4. Ranks (8)
  const zPos = [0, 1, 2, 3, 4, 5, 6, 7].map(i => ({ i, z: current[i] ? current[i].z : -1000 }));
  zPos.sort((a, b) => b.z - a.z);
  const ranks = new Array(8);
  zPos.forEach((item, index) => { ranks[item.i] = index + 1; });
  features.push(...ranks);

  // 5. Race Progress (1)
  features.push(Math.min(1.0, (stepCount * (1 / 120)) / 45.0));

  return features;
}

function runPredictions(features) {
  if (!xgboostWinnerModel || !features) return null;

  const probs = evaluateXGBoost(features, xgboostWinnerModel);
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
const OBSTACLE_SINK_TIME = 1.20;
const OBSTACLE_HOLD_TIME = 0.28;
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

  for (let i = 0; i < 7; i++) marbleRandom();

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

  // Lower linear damping so marbles keep enough momentum to climb inclines
  marbleBodies.forEach(b => { if (b) b.setLinearDamping(0.01); });

  const obsState = meta.obstacles.map(() => ({
    timer: obstacleRandom() * 1.0 + 1.0,
    state: 'waiting',
  }));

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
  const fallTimer = meta.marbles.map(() => 0);
  const stuckTimer = meta.marbles.map(() => 0);
  const kickCooldown = meta.marbles.map(() => 0);

  // ── Prediction state ──
  const predictionHistory = [];
  let lastPredictions = null;

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
    marbleStarts: meta.marbles.map((m) => marbleGridPos(m)),
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
        if (!body) continue;

        obs.timer -= PHYSICS_STEP;
        let sinkFrac = 0;

        switch (obs.state) {
          case 'waiting':
            if (obs.timer <= 0) { obs.state = 'sinking'; obs.timer = OBSTACLE_SINK_TIME; }
            sinkFrac = 0;
            break;
          case 'sinking':
            sinkFrac = 1 - obs.timer / OBSTACLE_SINK_TIME;
            if (obs.timer <= 0) { obs.state = 'sunk'; obs.timer = OBSTACLE_HOLD_TIME; sinkFrac = 1; }
            break;
          case 'sunk':
            sinkFrac = 1;
            if (obs.timer <= 0) { obs.state = 'rising'; obs.timer = OBSTACLE_SINK_TIME; }
            break;
          case 'rising':
            sinkFrac = obs.timer / OBSTACLE_SINK_TIME;
            if (obs.timer <= 0) { obs.state = 'waiting'; obs.timer = obstacleRandom() * 1.0 + 1.0; sinkFrac = 0; }
            break;
        }

        const s = sinkFrac * sinkFrac * (3 - 2 * sinkFrac);
        const sinkAmt = s * OBSTACLE_SINK_DEPTH;
        const oMeta = meta.obstacles[i];
        body.setNextKinematicTranslation({ x: oMeta.wx, y: oMeta.bodyWy - sinkAmt, z: oMeta.wz });
      }

      for (let i = 0; i < meta.blades.length; i++) {
        const blade = meta.blades[i];
        const body = bladeBodies[i];
        if (!body || !blade.baseQuat) continue;
        const bq = blade.baseQuat;
        _bladeRotOffset.setFromAxisAngle(_bladeAxis, physicsTimeTotal);
        _tempQ.set(bq.x, bq.y, bq.z, bq.w).multiply(_bladeRotOffset);
        body.setNextKinematicRotation({ x: _tempQ.x, y: _tempQ.y, z: _tempQ.z, w: _tempQ.w });
      }
    }

    physicsWorld.step();

    if (!gateIsOpen) {
      for (let i = 0; i < marbleBodies.length; i++) {
        const body = marbleBodies[i];
        const grid = marbleGridPos(meta.marbles[i]);
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
      
      if (isFinished) finished[i] = true;
      
      const vel = body.linvel();
      if (vel.y < -110 && Math.abs(vel.x) < 40 && Math.abs(vel.z) < 40) fallTimer[i] += PHYSICS_STEP;
      else fallTimer[i] = 0;
      if (fallTimer[i] > 1.0 || p.y < -50) {
        eliminated[i] = true;
        body.setTranslation({ x: 0, y: -1000, z: 0 }, true);
      }
    }

    const frame = buildFrame(physicsTick, gateIsOpen);

    if (lastPredictions && physicsTick % 60 === 0) {
      frame.predictions = lastPredictions;
    }

    yield { type: 'frame', frame };

    // Yield event loop every 60 frames (~0.5s of sim time) to stay responsive.
    if (physicsTick % 60 === 0) await yieldEventLoop();

    if (gateIsOpen && physicsTick % 60 === 0) {
      const snapFrame = meta.marbles.map((m, i) => {
        const body = marbleBodies[i];
        if (!body || eliminated[i] || finished[i]) return { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
        const p = body.translation();
        const v = body.linvel();
        return { x: p.x, y: p.y, z: p.z, vx: v.x, vy: v.y, vz: v.z };
      });
      predictionHistory.push(snapFrame);
      if (predictionHistory.length > 15) predictionHistory.shift();

      const features = buildPredictorFeatures(predictionHistory, physicsTick);
      lastPredictions = runPredictions(features);
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
