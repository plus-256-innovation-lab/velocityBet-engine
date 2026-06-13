import * as RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';
import fs from 'fs';
import { resolve } from 'path';

let rapierReady = false;

// ── In-memory store ──
let storedSnapshot = null;
let storedMetadata = null;

export async function storeSnapshot(snapshotBuffer, metadata) {
  storedSnapshot = snapshotBuffer;
  storedMetadata = metadata;
  fs.writeFileSync(resolve('..', 'track-snapshot.bin'), snapshotBuffer);
  fs.writeFileSync(resolve('..', 'track-metadata.json'), JSON.stringify(metadata));
  console.log(`[Engine] Snapshot stored: ${storedSnapshot.length} bytes, ${metadata.marbles?.length || 0} marbles`);
}

export async function ensureInitialized() {
  if (!rapierReady) {
    await RAPIER.init({});
    rapierReady = true;
    console.log('[Engine] Rapier initialized');
  }
}

// ── Autonomous Loading ──
export async function autoInitialize() {
  await ensureInitialized();
  const snapshotPath = resolve('..', 'track-snapshot.bin');
  const metaPath = resolve('..', 'track-metadata.json');
  
  if (fs.existsSync(snapshotPath) && fs.existsSync(metaPath)) {
    storedSnapshot = fs.readFileSync(snapshotPath);
    storedMetadata = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    console.log(`[Engine] Autonomous Init: Loaded snapshot (${storedSnapshot.length} bytes) and metadata.`);
  } else {
    throw new Error('Static assets track-snapshot.bin or track-metadata.json missing.');
  }
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

// ── Constants (matched to frontend) ──
const OBSTACLE_SINK_DEPTH = 14;
const OBSTACLE_SINK_TIME = 1.20;
const OBSTACLE_HOLD_TIME = 0.28;
const PHYSICS_STEP = 1 / 120;
const MARBLE_SETTLE_FRAMES = 90;

const _bladeAxis = new THREE.Vector3(1, 0, 0);
const _bladeRotOffset = new THREE.Quaternion();
const _tempQ = new THREE.Quaternion();
const _tempVec = new THREE.Vector3();

export async function* streamRace(seed) {
  if (!rapierReady) await autoInitialize();
  if (!storedSnapshot || !storedMetadata) await autoInitialize();

  const meta = storedMetadata;
  const seedNum = typeof seed === 'number' ? seed : (typeof seed === 'string' ? parseInt(seed, 10) : seed);
  
  const obstacleRandom = mulberry32(seedNum + 12345);
  const marbleRandom = mulberry32(seedNum + 54321);

  for (let i = 0; i < 7; i++) marbleRandom();

  const physicsWorld = RAPIER.World.restoreSnapshot(storedSnapshot);
  physicsWorld.gravity = { x: 0, y: -9.81 * 15.2, z: 0 };
  physicsWorld.timestep = PHYSICS_STEP;

  function getBody(handle) { return handle !== -1 ? physicsWorld.getRigidBody(handle) : null; }

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
  
  // physicsWorld.forEachCollider(collider => {
  //   const parent = collider.parent();
  //   if (parent && parent.bodyType() === RAPIER.RigidBodyType.Fixed) {
  //     collider.setRestitution(0.12);
  //     collider.setFriction(0.2);
  //   }
  // });

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

  yield { type: 'start', seed: seedNum, physicsStep: PHYSICS_STEP };

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

    if (!countdownStarted && !gateIsOpen && marbleBodies.length > 0) {
      if (physicsTick === 120) {
        countdownStarted = true;
        gateOpenTick = physicsTick + 420;
        impulseTick = gateOpenTick + 36;
      }
    }

    if (physicsTick === gateOpenTick) {
      gateIsOpen = true;
      if (gateBody) {
        gateBody.setNextKinematicTranslation({ x: 0, y: -1000, z: 0 });
      }
    }

    if (physicsTick === impulseTick) {
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
          true
        );
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

    const frame = {
      marbles: meta.marbles.map((m, i) => {
        const body = marbleBodies[i];
        if (!body) return null;
        const p = body.translation();
        const r = body.rotation();
        return [p.x, p.y, p.z, r.x, r.y, r.z, r.w, eliminated[i] ? 1 : 0, finished[i] ? 1 : 0];
      }),
      obstacles: meta.obstacles.map((obs, i) => { const body = obstacleBodies[i]; return body ? body.translation().y : 0; }),
      blades: meta.blades.map((b, i) => { const body = bladeBodies[i]; if (!body) return [0, 0, 0, 1]; const r = body.rotation(); return [r.x, r.y, r.z, r.w]; }),
      gateIsOpen: gateIsOpen
    };
    
    yield { type: 'frame', frame };
    
    // Yield to event loop occasionally to avoid blocking the server entirely
    if (physicsTick % 120 === 0) await new Promise(r => setTimeout(r, 0));
    
    let activeCount = 0;
    for (let i = 0; i < marbleBodies.length; i++) {
      if (!eliminated[i] && !finished[i]) activeCount++;
    }
    if (activeCount === 0) break;
  }

  physicsWorld.free();
  yield { type: 'end' };
}
