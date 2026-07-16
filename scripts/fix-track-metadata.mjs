/**
 * Remap Rapier rigid-body handles in track-metadata.json by matching snapshot body
 * positions. The committed metadata had handles serialized as ~0 (4.73e-321), so
 * deployed physics could not find marble bodies.
 */
import * as RAPIER from '@dimforge/rapier3d-compat';
import fs from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function dist3(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

function nearestBody(bodies, target, used, maxDist = 25) {
  let best = null;
  let bestD = Infinity;
  for (const body of bodies) {
    if (used.has(body.handle)) continue;
    const p = body.translation();
    const d = dist3(p, target);
    if (d < bestD) {
      bestD = d;
      best = body;
    }
  }
  if (!best || bestD > maxDist * maxDist) return null;
  used.add(best.handle);
  return best;
}

async function main() {
  await RAPIER.init();

  const snapshotPath = resolve(root, 'track-snapshot.bin');
  const metaPath = resolve(root, 'track-metadata.json');
  const snapshot = fs.readFileSync(snapshotPath);
  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));

  const world = RAPIER.World.restoreSnapshot(snapshot);
  world.gravity = { x: 0, y: -9.81 * 15.2, z: 0 };

  const bodies = [];
  world.forEachRigidBody((body) => bodies.push(body));
  console.log(`Loaded ${bodies.length} rigid bodies from snapshot`);

  const used = new Set();
  let fixed = 0;

  for (const m of meta.marbles) {
    const grid = { x: m.startPos.x, y: m.startPos.y, z: m.startPos.z - 3 };
    const body = nearestBody(bodies, grid, used, 8) ?? nearestBody(bodies, m.startPos, used, 8);
    if (!body) throw new Error(`No body found for marble at ${JSON.stringify(m.startPos)}`);
    m.handle = String(body.handle);
    fixed++;
  }

  for (const o of meta.obstacles) {
    const body = nearestBody(bodies, { x: o.wx, y: o.bodyWy, z: o.wz }, used, 12);
    if (!body) throw new Error(`No body found for obstacle at ${o.wx},${o.bodyWy},${o.wz}`);
    o.handle = String(body.handle);
    fixed++;
  }

  for (const b of meta.blades) {
    b._pending = true;
  }

  const remainingKinematic = bodies
    .filter((rb) => !used.has(rb.handle) && rb.isKinematic())
    .sort((a, b) => a.handle - b.handle);

  let ki = 0;
  for (const b of meta.blades) {
    const body = remainingKinematic[ki++];
    if (!body) throw new Error('Not enough kinematic bodies for blades');
    used.add(body.handle);
    b.handle = String(body.handle);
    delete b._pending;
    fixed++;
  }

  if (meta.handles?.gate != null && meta.handles.gate !== -1) {
    // Gate is kinematic near start line (~y=1133-1140).
    const gateCandidates = bodies
      .filter((b) => !used.has(b.handle) && b.isKinematic())
      .map((b) => ({ b, p: b.translation() }))
      .filter(({ p }) => p.y > 1120 && p.y < 1160 && p.z > -200 && p.z < -150)
      .sort((a, c) => c.p.y - a.p.y);
    const gate = gateCandidates[0]?.b ?? bodies.find((b) => !used.has(b.handle) && b.isKinematic());
    if (!gate) throw new Error('No gate body found');
    used.add(gate.handle);
    meta.handles.gate = String(gate.handle);
    fixed++;
  }

  fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2));
  console.log(`Fixed ${fixed} handles. Sample marble handle: ${meta.marbles[0].handle}`);
  world.free();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
