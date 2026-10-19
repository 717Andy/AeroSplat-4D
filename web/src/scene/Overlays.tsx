import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import type { GaussianCloud, OccupancyGrid } from "../codec";
import { interpolatePosition } from "../codec";
import { cameraCenter, frustumCorners, type RigCamera } from "../rig";

const RIG_COLORS: Record<string, string> = {
  front: "#3dcdc0",
  down: "#f0a202",
  port: "#7eb6ff",
  starboard: "#ff8a9a",
};

export function OccupancyBoxes({ grid }: { grid: OccupancyGrid }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const solids = useMemo(() => grid.cells.filter((cell) => cell.occupancy > grid.minProb), [grid]);
  const geometry = useMemo(() => new THREE.BoxGeometry(1, 1, 1), []);
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0.28,
        depthWrite: false,
        toneMapped: false,
      }),
    [],
  );
  useLayoutEffect(() => {
    const current = mesh.current;
    if (!current) return;
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    solids.forEach((cell, index) => {
      dummy.position.set(
        grid.origin[0] + (cell.i + 0.5) * grid.voxelSize,
        grid.origin[1] + (cell.j + 0.5) * grid.voxelSize,
        grid.origin[2] + (cell.k + 0.5) * grid.voxelSize,
      );
      dummy.scale.setScalar(grid.voxelSize * 0.9);
      dummy.rotation.set(0, 0, 0);
      dummy.updateMatrix();
      current.setMatrixAt(index, dummy.matrix);
      color.set(cell.wire > 0.5 ? "#3dcdc0" : "#e07a3d");
      current.setColorAt(index, color);
    });
    current.instanceMatrix.needsUpdate = true;
    if (current.instanceColor) current.instanceColor.needsUpdate = true;
  }, [grid, solids]);
  if (solids.length === 0) return null;
  return (
    <instancedMesh
      ref={mesh}
      args={[geometry, material, solids.length]}
      frustumCulled={false}
      renderOrder={2}
    />
  );
}

export function WireLines({ grid }: { grid: OccupancyGrid }) {
  const geometry = useMemo(() => {
    const wires = grid.cells.filter((cell) => cell.wire > grid.minProb);
    const positions = new Float32Array(wires.length * 6);
    const half = 0.5 * grid.voxelSize * Math.sqrt(3);
    wires.forEach((cell, index) => {
      const center = [
        grid.origin[0] + (cell.i + 0.5) * grid.voxelSize + cell.offset[0] * grid.voxelSize,
        grid.origin[1] + (cell.j + 0.5) * grid.voxelSize + cell.offset[1] * grid.voxelSize,
        grid.origin[2] + (cell.k + 0.5) * grid.voxelSize + cell.offset[2] * grid.voxelSize,
      ];
      const length = Math.hypot(cell.direction[0], cell.direction[1], cell.direction[2]) || 1;
      const dir = cell.direction.map((value) => value / length) as [number, number, number];
      const base = index * 6;
      positions[base] = center[0] - dir[0] * half;
      positions[base + 1] = center[1] - dir[1] * half;
      positions[base + 2] = center[2] - dir[2] * half;
      positions[base + 3] = center[0] + dir[0] * half;
      positions[base + 4] = center[1] + dir[1] * half;
      positions[base + 5] = center[2] + dir[2] * half;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [grid]);
  if (grid.cells.every((cell) => cell.wire <= grid.minProb)) return null;
  return (
    <lineSegments geometry={geometry} frustumCulled={false} renderOrder={3}>
      <lineBasicMaterial color="#b8fff4" />
    </lineSegments>
  );
}

export function VelocityArrows({
  cloud,
  alpha,
  dt,
}: {
  cloud: GaussianCloud;
  alpha: number;
  dt: number;
}) {
  const positions = useMemo(() => new Float32Array(400 * 6), []);
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [positions]);
  useLayoutEffect(() => {
    const moving: number[] = [];
    for (let i = 0; i < cloud.count; i += 1) {
      const speed = Math.hypot(cloud.velocity[i * 3], cloud.velocity[i * 3 + 1], cloud.velocity[i * 3 + 2]);
      if (cloud.valid[i] && speed > 0.2) moving.push(i);
    }
    const shown = moving.slice(0, 400);
    const point = { x: 0, y: 0, z: 0 };
    shown.forEach((index, slot) => {
      interpolatePosition(cloud.xyz, cloud.velocity, index, alpha, dt, point);
      const base = slot * 6;
      positions[base] = point.x;
      positions[base + 1] = point.y;
      positions[base + 2] = point.z;
      positions[base + 3] = point.x + cloud.velocity[index * 3] * 0.45;
      positions[base + 4] = point.y + cloud.velocity[index * 3 + 1] * 0.45;
      positions[base + 5] = point.z + cloud.velocity[index * 3 + 2] * 0.45;
    });
    for (let i = shown.length * 6; i < positions.length; i += 1) positions[i] = 0;
    geometry.attributes.position.needsUpdate = true;
    geometry.setDrawRange(0, shown.length * 2);
  }, [alpha, cloud, dt, geometry, positions]);
  return (
    <lineSegments geometry={geometry} frustumCulled={false} renderOrder={4}>
      <lineBasicMaterial color="#ff8a3d" />
    </lineSegments>
  );
}

export function CorridorLine({
  points,
  warned,
}: {
  points: [number, number, number][];
  warned: boolean;
}) {
  const geometry = useMemo(() => {
    const positions = new Float32Array(Math.max(points.length - 1, 0) * 6);
    for (let index = 0; index < points.length - 1; index += 1) {
      const a = points[index];
      const b = points[index + 1];
      const base = index * 6;
      positions[base] = a[0];
      positions[base + 1] = a[1];
      positions[base + 2] = a[2];
      positions[base + 3] = b[0];
      positions[base + 4] = b[1];
      positions[base + 5] = b[2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [points]);
  if (points.length < 2) return null;
  return (
    <lineSegments geometry={geometry} frustumCulled={false} renderOrder={4}>
      <lineBasicMaterial color={warned ? "#ff5a5a" : "#9be7a4"} />
    </lineSegments>
  );
}

export function CameraRig({ cameras }: { cameras: RigCamera[] }) {
  return (
    <group>
      {cameras.map((camera) => (
        <CameraFrustum key={camera.name} camera={camera} />
      ))}
    </group>
  );
}

function CameraFrustum({ camera }: { camera: RigCamera }) {
  const color = RIG_COLORS[camera.name] ?? "#d5e6dc";
  const segments = useMemo(() => {
    const near = frustumCorners(camera, 0.35);
    const far = frustumCorners(camera, 2.4);
    const center = cameraCenter(camera.R, camera.t);
    const positions: number[] = [];
    const push = (a: number[], b: number[]) => positions.push(...a, ...b);
    for (let i = 0; i < 4; i += 1) {
      push(near[i], near[(i + 1) % 4]);
      push(far[i], far[(i + 1) % 4]);
      push(center, far[i]);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return geo;
  }, [camera]);
  const plane = useMemo(() => planeGeometry(camera, 1.35), [camera]);
  const texture = useMemo(() => {
    const loader = new THREE.TextureLoader();
    const map = loader.load(camera.imageUrl);
    map.colorSpace = THREE.SRGBColorSpace;
    return map;
  }, [camera.imageUrl]);
  return (
    <group>
      <lineSegments geometry={segments} frustumCulled={false} renderOrder={5}>
        <lineBasicMaterial color={color} />
      </lineSegments>
      <mesh geometry={plane} frustumCulled={false} renderOrder={5}>
        <meshBasicMaterial map={texture} transparent opacity={0.92} side={THREE.DoubleSide} toneMapped={false} />
      </mesh>
    </group>
  );
}

function planeGeometry(camera: RigCamera, depth: number) {
  const corners = frustumCorners(camera, depth);
  const positions = new Float32Array(18);
  const order = [0, 1, 2, 0, 2, 3];
  order.forEach((corner, index) => {
    positions[index * 3] = corners[corner][0];
    positions[index * 3 + 1] = corners[corner][1];
    positions[index * 3 + 2] = corners[corner][2];
  });
  const uvs = new Float32Array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  return geo;
}
