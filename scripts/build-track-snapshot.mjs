/**
 * Build Rapier physics snapshot + metadata directly from track-v1.glb.
 * Used in the engine Docker image so frontend upload is not required.
 *
 * Usage: node scripts/build-track-snapshot.mjs [path/to/track-v1.glb]
 */
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import * as RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import draco3d from 'draco3dgltf';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const OBSTACLE_NAME_RE = /obstacle|peg|bump|pin|stud|post|knob|pillar|bumper|rivet/i;
const DECOR_SKIP_RE =
  /pavillion|pavilion|stadium|seat|chair|bench|stand|banner|pasted|sphere|icosphere|spectator|crowd|audience/i;
const MARBLE_COUNT = 8;
const GRAVITY = { x: 0.0, y: -9.81 * 15, z: 0.0 };

function nodeWorldMatrix(node) {
  const m = new THREE.Matrix4();
  const chain = [];
  for (let n = node; n; n = n.getParent ? n.getParent() : null) chain.unshift(n);
  for (const n of chain) {
    const t = n.getTranslation?.() ?? [0, 0, 0];
    const r = n.getRotation?.() ?? [0, 0, 0, 1];
    const s = n.getScale?.() ?? [1, 1, 1];
    const local = new THREE.Matrix4().compose(
      new THREE.Vector3(t[0], t[1], t[2]),
      new THREE.Quaternion(r[0], r[1], r[2], r[3]),
      new THREE.Vector3(s[0], s[1], s[2]),
    );
    m.multiply(local);
  }
  return m;
}

function isObstacleName(name, parentName = '') {
  return OBSTACLE_NAME_RE.test(name) || OBSTACLE_NAME_RE.test(parentName);
}

function isBladeName(name, parentName = '') {
  const n = (name || '').toLowerCase();
  const p = (parentName || '').toLowerCase();
  return n.includes('blade') || p.includes('blade');
}

function isCrowdMarbleName(name, parentName = '') {
  return /sphere|icosphere|uvsphere|marble|ball|spectator|crowd/i.test(name || '') ||
    /sphere|icosphere|uvsphere|marble|ball|spectator|crowd/i.test(parentName || '');
}

function isGateName(name) {
  return name === 'gate' || name === 'Kairos-Body-5_2';
}

function expandBB(bb, { name, parentName }) {
  const nameLower = (name || '').replace(/_|-/g, ' ').toLowerCase();
  const parentLower = (parentName || '').replace(/_|-/g, ' ').toLowerCase();
  return {
    isBend: nameLower.includes('bend') || parentLower.includes('bend'),
    isFinish: /finish/i.test(name) || /finish/i.test(parentName),
    isLastRun:
      (nameLower.includes('last') && nameLower.includes('run')) ||
      (parentLower.includes('last') && parentLower.includes('run')),
  };
}

function unionBox(target, box) {
  if (!target) return box.clone();
  return target.union(box);
}

function meshGeometryFromPrimitive(prim, worldMatrix) {
  const pos = prim.getAttribute('POSITION');
  if (!pos) return null;
  const arr = pos.getArray();
  const vertices = new Float32Array(arr.length);
  vertices.set(arr);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  const idx = prim.getIndices();
  if (idx) {
    geo.setIndex(Array.from(idx.getArray()));
  }
  geo.applyMatrix4(worldMatrix);
  return geo;
}

function colliderFromGeometry(world, geo, { friction = 0.2, restitution = 0.12, kinematic = false, translation, rotation } = {}) {
  const posAttr = geo.attributes.position;
  const vertices = new Float32Array(posAttr.array);
  let indices;
  if (geo.index) {
    indices = new Uint32Array(geo.index.array);
  } else {
    const n = vertices.length / 3;
    indices = new Uint32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;
  }

  const bodyDesc = kinematic
    ? RAPIER.RigidBodyDesc.kinematicPositionBased()
    : RAPIER.RigidBodyDesc.fixed();
  if (translation) bodyDesc.setTranslation(translation.x, translation.y, translation.z);
  if (rotation) bodyDesc.setRotation(rotation);

  const body = world.createRigidBody(bodyDesc);
  world.createCollider(
    RAPIER.ColliderDesc.trimesh(vertices, indices).setFriction(friction).setRestitution(restitution),
    body,
  );
  return body;
}

async function loadDocument(glbPath) {
  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      'draco3d.decoder': await draco3d.createDecoderModule(),
    });
  return io.read(glbPath);
}

async function main() {
  const glbPath = path.resolve(process.argv[2] || path.join(ROOT, 'track-v1.glb'));
  if (!fs.existsSync(glbPath)) {
    throw new Error(`Missing track GLB at ${glbPath}. Run: node scripts/prepare-track.mjs`);
  }

  console.log(`[snapshot] Loading ${glbPath} (${fs.statSync(glbPath).size} bytes)`);
  await RAPIER.init();
  const world = new RAPIER.World(GRAVITY);

  const document = await loadDocument(glbPath);
  const root = document.getRoot();

  const meshes = [];
  for (const scene of root.listScenes()) {
    scene.traverse((node) => {
      const mesh = node.getMesh();
      if (!mesh) return;
      const parent = node.getParentNode?.() || node.getParent?.() || null;
      const parentName = parent?.getName?.() || '';
      const worldMatrix = nodeWorldMatrix(node);
      for (const prim of mesh.listPrimitives()) {
        const geo = meshGeometryFromPrimitive(prim, worldMatrix);
        if (!geo) continue;
        meshes.push({
          name: node.getName() || mesh.getName() || '',
          parentName,
          geo,
          worldMatrix,
        });
      }
    });
  }

  console.log(`[snapshot] Found ${meshes.length} mesh primitives`);

  let gateMesh = null;
  const obstacleMeshes = [];
  const bladeMeshes = [];
  let bendBB = null;
  let finishPlaneBB = null;
  let lastRunBB = null;
  const seenStaticBases = new Set();
  let skippedDupStatic = 0;

  for (const m of meshes) {
    const tags = expandBB(null, m);
    m.geo.computeBoundingBox();
    const bb = m.geo.boundingBox.clone();

    if (tags.isBend) {
      bb.expandByScalar(2.0);
      bendBB = unionBox(bendBB, bb);
    }
    if (tags.isFinish) {
      const f = bb.clone();
      f.expandByVector(new THREE.Vector3(0.5, 0.0, 0.5));
      f.min.y -= 2.0;
      finishPlaneBB = unionBox(finishPlaneBB, f);
    }
    if (tags.isLastRun) {
      bb.expandByScalar(20.0);
      lastRunBB = unionBox(lastRunBB, bb);
    }

    if (isGateName(m.name)) {
      gateMesh = m;
      continue;
    }
    if (isObstacleName(m.name, m.parentName)) {
      obstacleMeshes.push(m);
      continue;
    }
    if (isBladeName(m.name, m.parentName)) {
      bladeMeshes.push(m);
      continue;
    }

    // Visual-only decor — including these as trimeshes ballooned restoreSnapshot to ~30–40s/race.
    if (
      DECOR_SKIP_RE.test(m.name) ||
      DECOR_SKIP_RE.test(m.parentName) ||
      isCrowdMarbleName(m.name, m.parentName)
    ) {
      continue;
    }

    // Blender duplicate suffixes (.001/.002) — keep one static collider per base name.
    // Pegs/blades are handled above and must NOT be collapsed this way.
    const baseName = (m.name || '').replace(/\.\d+$/, '');
    const vertCount = m.geo.attributes.position?.count || 0;
    const dedupeKey = `${baseName}::${vertCount}`;
    if (seenStaticBases.has(dedupeKey)) {
      skippedDupStatic++;
      continue;
    }
    seenStaticBases.add(dedupeKey);

    colliderFromGeometry(world, m.geo);
  }

  console.log(
    `[snapshot] static built; obstacles=${obstacleMeshes.length} blades=${bladeMeshes.length} gate=${gateMesh ? 'yes' : 'no'} skippedDupStatic=${skippedDupStatic}`,
  );

  const obstacles = [];
  for (const m of obstacleMeshes) {
    m.geo.computeBoundingBox();
    const bbSize = m.geo.boundingBox.getSize(new THREE.Vector3());
    const localCenter = m.geo.boundingBox.getCenter(new THREE.Vector3());
    // geometry already in world space
    const wCenter = localCenter;
    const halfH = bbSize.y / 2;
    const halfR = Math.max(bbSize.x, bbSize.z) / 2;
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    m.worldMatrix.decompose(pos, quat, scale);

    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased()
        .setTranslation(wCenter.x, wCenter.y, wCenter.z)
        .setRotation({ x: quat.x, y: quat.y, z: quat.z, w: quat.w }),
    );
    world.createCollider(
      RAPIER.ColliderDesc.cylinder(halfH, halfR).setFriction(0.3).setRestitution(0.1),
      body,
    );
    obstacles.push({ handle: String(body.handle), wx: wCenter.x, bodyWy: wCenter.y, wz: wCenter.z });
  }

  const blades = [];
  for (const m of bladeMeshes) {
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    m.worldMatrix.decompose(pos, quat, scale);

    // geometry already world-transformed; rebuild local-scaled verts around body origin
    const clone = m.geo.clone();
    const inv = new THREE.Matrix4().copy(m.worldMatrix).invert();
    clone.applyMatrix4(inv);
    clone.scale(scale.x, scale.y, scale.z);

    const body = colliderFromGeometry(world, clone, {
      kinematic: true,
      friction: 0.2,
      restitution: 0.1,
      translation: pos,
      rotation: { x: quat.x, y: quat.y, z: quat.z, w: quat.w },
    });
    blades.push({
      handle: String(body.handle),
      baseQuat: { x: quat.x, y: quat.y, z: quat.z, w: quat.w },
    });
  }

  let gateBody = null;
  let gateBB = null;
  let gateHingeZ = null;
  let trackForwardDir = { x: 0, y: 0, z: 1 };

  if (gateMesh) {
    gateMesh.geo.computeBoundingBox();
    gateBB = gateMesh.geo.boundingBox.clone();
    const bbSize = gateBB.getSize(new THREE.Vector3());
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    gateMesh.worldMatrix.decompose(pos, quat, scale);

    const pivotPos = new THREE.Vector3(pos.x, gateBB.min.y, gateBB.min.z);
    gateHingeZ = pivotPos.z;
    const closedRotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
    const localCenter = pos.clone().sub(pivotPos);
    localCenter.applyQuaternion(closedRotation);
    const closedCenter = pivotPos.clone().add(localCenter);
    const closedMeshQuat = closedRotation.clone().multiply(quat);

    const halfX = bbSize.x * 0.5 * 1.2;
    const halfY = bbSize.y * 0.5;
    const halfZ = bbSize.z * 0.5;

    gateBody = world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased()
        .setTranslation(closedCenter.x, closedCenter.y, closedCenter.z)
        .setRotation({ x: closedMeshQuat.x, y: closedMeshQuat.y, z: closedMeshQuat.z, w: closedMeshQuat.w }),
    );
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(halfX, halfY, halfZ).setFriction(0.4).setRestitution(0.05),
      gateBody,
    );
  } else {
    console.warn('[snapshot] No gate mesh found — marbles may be misplaced');
    gateBB = new THREE.Box3(new THREE.Vector3(-100, 1100, -200), new THREE.Vector3(0, 1150, -150));
  }

  // Marbles (deterministic lanes — no shuffle)
  const worldCenter = gateBB.getCenter(new THREE.Vector3());
  const worldSize = gateBB.getSize(new THREE.Vector3());
  const marbleR = Math.max(1.3, worldSize.x / 34);
  const hingeZ = gateHingeZ ?? gateBB.min.z;
  const spawnZ = hingeZ - marbleR * 3.5 - 3;
  const spawnY = gateBB.max.y + marbleR * 1.5;
  trackForwardDir = { x: 0, y: 0, z: Math.sign(worldCenter.z - spawnZ) || 1 };

  const floorHX = worldSize.x * 0.6;
  const floorHY = 0.2;
  const floorHZ = worldSize.x * 0.6;
  const shiftBackZ = trackForwardDir.z * (floorHZ * 0.4);
  const floorBody = world.createRigidBody(
    RAPIER.RigidBodyDesc.fixed().setTranslation(worldCenter.x, spawnY - marbleR * 2.5, spawnZ - shiftBackZ),
  );
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(floorHX, floorHY, floorHZ).setFriction(0.1).setRestitution(0.05),
    floorBody,
  );

  const totalSpan = worldSize.x * 0.65;
  const laneSpacing = MARBLE_COUNT > 1 ? totalSpan / (MARBLE_COUNT - 1) : 0;
  const startX = worldCenter.x - totalSpan / 2;
  const marbles = [];
  for (let i = 0; i < MARBLE_COUNT; i++) {
    const x = startX + i * laneSpacing;
    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(x, spawnY, spawnZ)
        .setLinearDamping(0.05)
        .setAngularDamping(0.05),
    );
    world.createCollider(
      RAPIER.ColliderDesc.ball(marbleR).setMass(1.0).setFriction(0.2).setRestitution(0.12),
      body,
    );
    marbles.push({
      handle: String(body.handle),
      radius: marbleR,
      startPos: { x, y: spawnY, z: spawnZ },
    });
  }

  world.step();
  const snapshot = world.takeSnapshot();
  const raw = Buffer.from(snapshot);
  const gz = zlib.gzipSync(raw, { level: 6 });
  const snapPath = path.join(ROOT, 'track-snapshot.bin.gz');
  const metaPath = path.join(ROOT, 'track-metadata.json');

  const boxToJson = (b) =>
    b
      ? { min: { x: b.min.x, y: b.min.y, z: b.min.z }, max: { x: b.max.x, y: b.max.y, z: b.max.z } }
      : null;

  const metadata = {
    marbles,
    obstacles,
    blades,
    handles: { gate: gateBody ? String(gateBody.handle) : -1 },
    trackForwardDir,
    bendBB: boxToJson(bendBB),
    glassyExtraBB: null,
    finishPlaneBB: boxToJson(finishPlaneBB),
    lastRunBB: boxToJson(lastRunBB),
  };

  fs.writeFileSync(snapPath, gz);
  fs.writeFileSync(metaPath, `${JSON.stringify(metadata, null, 2)}\n`);

  console.log(`[snapshot] Wrote ${snapPath} (${gz.length} bytes gzip, ${raw.length} raw)`);
  console.log(
    `[snapshot] Wrote ${metaPath} (obstacles=${obstacles.length}, blades=${blades.length}, marbles=${marbles.length})`,
  );
}

main().catch((err) => {
  console.error('[snapshot] FAILED', err);
  process.exit(1);
});
