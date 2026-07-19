import * as RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';
import fs from 'fs';
import { resolve, join } from 'path';
import zlib from 'zlib';

// ── Constants ──
const PHYSICS_STEP = 1 / 120;
const OBSTACLE_SINK_DEPTH = 14;
const OBSTACLE_SINK_TIME = 2.0;
const OBSTACLE_RISE_TIME = 2.0;
const OBSTACLE_HOLD_TIME = 0.25;
const OBSTACLE_WAIT_TIME = 0.25;
const OBSTACLE_HOLD_TIME_LONG = 2.5;
const OBSTACLE_WAIT_TIME_LONG = 4.0;
const SAMPLE_INTERVAL = 30;
const MARBLE_NAMES = ['Yellow', 'White', 'Cyan', 'Green', 'Red', 'Blue', 'Orange', 'Purple'];

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

async function loadTrack() {
  await RAPIER.init({});
  console.log('[Init] Rapier ready');

  const snapshotGzPath = resolve('..', 'track-snapshot.bin.gz');
  const snapshotPath = resolve('..', 'track-snapshot.bin');
  const metaPath = resolve('..', 'track-metadata.json');

  let rawBuffer;
  if (fs.existsSync(snapshotGzPath)) {
    rawBuffer = fs.readFileSync(snapshotGzPath);
    rawBuffer = zlib.gunzipSync(rawBuffer);
  } else if (fs.existsSync(snapshotPath)) {
    rawBuffer = fs.readFileSync(snapshotPath);
  } else {
    throw new Error('Track snapshot not found');
  }
  const metadata = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
  console.log(`[Init] Loaded snapshot (${rawBuffer.length} bytes) and metadata`);
  return { snapshot: rawBuffer, metadata };
}

function applyWorldDefaults(world) {
  world.gravity = { x: 0, y: -9.81 * 15.2, z: 0 };
  world.timestep = PHYSICS_STEP;
}

let cachedCleanSnapshot = null;

function runRace(metadata, seed) {
  const meta = metadata;
  const obstacleRandom = mulberry32(seed + 12345);
  const marbleRandom = mulberry32(seed + 54321);
  for (let i = 0; i < 7; i++) marbleRandom();

  // Restore from cached clean snapshot (fast) instead of full 500MB snapshot
  const physicsWorld = RAPIER.World.restoreSnapshot(cachedCleanSnapshot);
  applyWorldDefaults(physicsWorld);

  function getBody(handle) { return handle !== -1 ? physicsWorld.getRigidBody(handle) : null; }

  const marbleBodies = meta.marbles.map(m => getBody(m.handle));
  const obstacleBodies = meta.obstacles.map(o => getBody(o.handle));
  const bladeBodies = meta.blades.map(b => getBody(b.handle));
  const gateBody = getBody(meta.handles.gate);

  marbleBodies.forEach(b => { if (b) { b.collider(0).setRestitution(0.12); b.setLinearDamping(0.01); } });
  obstacleBodies.forEach(b => { if (b) b.collider(0).setRestitution(0.1); });
  bladeBodies.forEach(b => { if (b) b.collider(0).setRestitution(0.12); });
  if (gateBody) gateBody.collider(0).setRestitution(0.05);

  function obstacleInBB(oMeta, bb, yPad = 0) {
    if (!bb?.min || !bb?.max || !oMeta) return false;
    return (
      oMeta.wx >= bb.min.x && oMeta.wx <= bb.max.x &&
      oMeta.bodyWy >= bb.min.y - yPad && oMeta.bodyWy <= bb.max.y + yPad &&
      oMeta.wz >= bb.min.z && oMeta.wz <= bb.max.z
    );
  }
  function isLongStayObstacle(oMeta) {
    return (
      obstacleInBB(oMeta, meta.finishPlaneBB, 25) ||
      obstacleInBB(oMeta, meta.lastRunBB, 10)
    );
  }
  function obstacleWaitTime(oMeta) {
    return isLongStayObstacle(oMeta) ? OBSTACLE_WAIT_TIME_LONG : OBSTACLE_WAIT_TIME;
  }
  function obstacleHoldTime(oMeta) {
    return isLongStayObstacle(oMeta) ? OBSTACLE_HOLD_TIME_LONG : OBSTACLE_HOLD_TIME;
  }

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
  const maxTicks = 120 * 300;
  const eliminated = meta.marbles.map(() => false);
  const finished = meta.marbles.map(() => false);
  const fallTimer = meta.marbles.map(() => 0);
  const stuckTimer = meta.marbles.map(() => 0);
  const kickCooldown = meta.marbles.map(() => 0);
  const samples = [];
  let winner = -1;

  let ticksSinceWinner = 0;

  while (physicsTick < maxTicks) {
    const activeCount = finished.filter(f => !f).filter((_, i) => !eliminated[i]).length;
    if (activeCount === 0) break;
    if (ticksSinceWinner > 120) break; // Stop 1 second after winner

    physicsTick++;
    physicsTimeTotal += PHYSICS_STEP;

    // Obstacles
    if (gateIsOpen) {
      for (let i = 0; i < meta.obstacles.length; i++) {
        const obs = obsState[i];
        const body = obstacleBodies[i];
        const oMeta = meta.obstacles[i];
        if (!body || !oMeta) continue;
        obs.timer -= PHYSICS_STEP;
        let sinkFrac = 0;
        switch (obs.state) {
          case 'waiting': if (obs.timer <= 0) { obs.state = 'sinking'; obs.timer = OBSTACLE_SINK_TIME; } break;
          case 'sinking': sinkFrac = 1 - obs.timer / OBSTACLE_SINK_TIME; if (obs.timer <= 0) { obs.state = 'sunk'; obs.timer = obstacleHoldTime(oMeta); sinkFrac = 1; } break;
          case 'sunk': sinkFrac = 1; if (obs.timer <= 0) { obs.state = 'rising'; obs.timer = OBSTACLE_RISE_TIME; } break;
          case 'rising': sinkFrac = obs.timer / OBSTACLE_RISE_TIME; if (obs.timer <= 0) { obs.state = 'waiting'; obs.timer = obstacleWaitTime(oMeta); sinkFrac = 0; } break;
        }
        const s = sinkFrac * sinkFrac * (3 - 2 * sinkFrac);
        body.setNextKinematicTranslation({ x: oMeta.wx, y: oMeta.bodyWy - s * OBSTACLE_SINK_DEPTH, z: oMeta.wz });
      }
      for (let i = 0; i < meta.blades.length; i++) {
        const blade = meta.blades[i];
        const body = bladeBodies[i];
        if (!body || !blade.baseQuat) continue;
        const randomPhase = blade.mesh ? (blade.mesh.id % 10) * 1.2 : i * 1.5;
        _bladeRotOffset.setFromAxisAngle(_bladeAxis, physicsTimeTotal + randomPhase);
        _tempQ.set(blade.baseQuat.x, blade.baseQuat.y, blade.baseQuat.z, blade.baseQuat.w).multiply(_bladeRotOffset);
        body.setNextKinematicRotation({ x: _tempQ.x, y: _tempQ.y, z: _tempQ.z, w: _tempQ.w });
      }
    }

    physicsWorld.step();

    // Bend boost
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
          if (speed > 0.5 && speed < 18) body.setLinvel({ x: vel.x * 2.30, y: vel.y, z: vel.z * 2.30 }, true);
        }
      }
    }

    // Glassy boost
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
            if (speed > 0.2) body.setLinvel({ x: (vel.x / speed) * 25.0, y: vel.y, z: (vel.z / speed) * 25.0 }, true);
            else body.setLinvel({ x: 0.968 * 25.0, y: vel.y, z: 0.252 * 25.0 }, true);
          }
        }
      }
    }

    // Stuck kick
    if (gateIsOpen) {
      for (let i = 0; i < marbleBodies.length; i++) {
        const body = marbleBodies[i];
        if (!body || eliminated[i] || finished[i]) continue;
        if (kickCooldown[i] > 0) { kickCooldown[i]--; continue; }
        const vel = body.linvel();
        const speed = Math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        const r = meta.marbles[i]?.radius || 1;
        if (speed < r * 0.4) {
          stuckTimer[i]++;
          if (stuckTimer[i] >= 90) {
            const kick = r * 9;
            const angle = marbleRandom() * Math.PI * 2;
            body.applyImpulse({ x: Math.cos(angle) * kick, y: kick * 0.4, z: Math.sin(angle) * kick }, true);
            stuckTimer[i] = 0;
            kickCooldown[i] = 120;
          }
        } else { stuckTimer[i] = 0; }
      }
    }

    // Gate timing
    if (!countdownStarted && !gateIsOpen && marbleBodies.length > 0) {
      if (physicsTick === 120) { countdownStarted = true; gateOpenTick = physicsTick + 420; impulseTick = gateOpenTick + 36; }
    }
    if (physicsTick === gateOpenTick) {
      gateIsOpen = true;
      if (gateBody) gateBody.setNextKinematicTranslation({ x: 0, y: -1000, z: 0 });
    }
    if (physicsTick === impulseTick) {
      for (let i = 0; i < marbleBodies.length; i++) {
        const body = marbleBodies[i];
        if (!body || eliminated[i]) continue;
        const marbleR = meta.marbles[i]?.radius || 1;
        const forwardImpulse = marbleR * 20.0;
        const jitter = (marbleRandom() - 0.5) * marbleR * 3.0;
        body.applyImpulse({ x: trackForwardDir.x * forwardImpulse + jitter, y: 0, z: trackForwardDir.z * forwardImpulse + jitter }, true);
      }
    }

    // Finish / elimination detection
    for (let i = 0; i < marbleBodies.length; i++) {
      const body = marbleBodies[i];
      if (eliminated[i] || finished[i] || !body) continue;
      const p = body.translation();
      let isFinished = false;
      if (meta.finishPlaneBB) {
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
        if (winner === -1) winner = i;
      }
      const vel = body.linvel();
      if (vel.y < -110 && Math.abs(vel.x) < 40 && Math.abs(vel.z) < 40) fallTimer[i] += PHYSICS_STEP;
      else fallTimer[i] = 0;
      if (fallTimer[i] > 1.0 || p.y < -50) {
        eliminated[i] = true;
        body.setTranslation({ x: 0, y: -1000, z: 0 }, true);
      }
    }
    if (winner >= 0) ticksSinceWinner++;

    // Sample
    if (gateIsOpen && physicsTick % SAMPLE_INTERVAL === 0) {
      const sample = [];
      for (let i = 0; i < 8; i++) {
        const body = marbleBodies[i];
        if (!body || eliminated[i]) { sample.push(0, 0, 0, 0, 0, 0); continue; }
        const p = body.translation();
        const v = body.linvel();
        sample.push(p.x, p.y, p.z, v.x, v.y, v.z);
      }
      samples.push({ tick: physicsTick, data: sample });
    }
  }

  physicsWorld.free();
  return { winner, samples, totalTicks: physicsTick };
}

// ── Main ──
const CSV_PATH = resolve('marble_race_data_v1.csv');

const header = [
  'race_id', 'seed', 'tick', 'winner',
  ...MARBLE_NAMES.flatMap(n => [`${n}_x`, `${n}_y`, `${n}_z`, `${n}_vx`, `${n}_vy`, `${n}_vz`])
].join(',');

async function main() {
  console.log(`[Collect] Loading track...`);
  const { snapshot, metadata } = await loadTrack();

  // Restore once from full snapshot, then cache a clean snapshot for fast resets
  console.log(`[Collect] Restoring from full snapshot (one-time)...`);
  const t0 = Date.now();
  let tempWorld = RAPIER.World.restoreSnapshot(snapshot);
  applyWorldDefaults(tempWorld);
  cachedCleanSnapshot = tempWorld.takeSnapshot();
  tempWorld.free();
  tempWorld = null;
  console.log(`[Collect] Clean snapshot cached in ${((Date.now() - t0) / 1000).toFixed(1)}s (${cachedCleanSnapshot.length} bytes)`);

  console.log(`[Collect] Track ready. Running races until stopped (Ctrl+C)...\n`);

  const startTime = Date.now();
  const winCounts = {};
  const writeHeader = !fs.existsSync(CSV_PATH) || fs.statSync(CSV_PATH).size === 0;
  const fd = fs.openSync(CSV_PATH, 'a');
  if (writeHeader) fs.writeSync(fd, header + '\n');

  for (let i = 0; ; i++) {
    const seed = Math.floor(Math.random() * 0xFFFFFFFF);
    const result = runRace(metadata, seed);
    const winnerName = result.winner >= 0 ? MARBLE_NAMES[result.winner] : 'none';
    winCounts[winnerName] = (winCounts[winnerName] || 0) + 1;

    for (const s of result.samples) {
      const vals = [i + 1, seed, s.tick, winnerName, ...s.data];
      fs.writeSync(fd, vals.join(',') + '\n');
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const rate = ((i + 1) / (Date.now() - startTime) * 1000).toFixed(1);
    process.stdout.write(`\r[Race ${i + 1}] Winner: ${winnerName.padEnd(8)} | ${rate} races/sec | ${elapsed}s`);
  }

  process.on('SIGINT', () => {
    fs.closeSync(fd);
    const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
    const total = Object.values(winCounts).reduce((a, b) => a + b, 0);
    console.log(`\n\n[Collect] Stopped. ${total} races in ${totalTime}s`);
    console.log(`[Collect] CSV: ${CSV_PATH}`);
    console.log(`\nWin distribution:`);
    for (const [name, count] of Object.entries(winCounts).sort((a, b) => b[1] - a[1])) {
      console.log(`  ${name.padEnd(10)} ${count.toString().padStart(5)} (${(count / total * 100).toFixed(1)}%)`);
    }
    process.exit(0);
  });
}

main().catch(err => { console.error('[Collect] Fatal:', err); process.exit(1); });
