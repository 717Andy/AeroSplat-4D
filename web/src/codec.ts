/** AGS1 / AOC1 readers. Little-endian, matching `server/codec.py`. */

export const GAUSSIAN_HEADER_BYTES = 64;
export const OCCUPANCY_HEADER_BYTES = 64;
export const CHANNELS = 62;
export const SH_COEFFS = 16;

export type GaussianCloud = {
  batch: number;
  count: number;
  timestamp: number;
  dt: number;
  sourceSun: [number, number, number];
  xyz: Float32Array;
  scale: Float32Array;
  quat: Float32Array;
  opacity: Float32Array;
  sh: Float32Array;
  velocity: Float32Array;
  valid: Uint8Array;
  wireProb: Float32Array;
};

export type OccupancyCell = {
  i: number;
  j: number;
  k: number;
  occupancy: number;
  wire: number;
  direction: [number, number, number];
  offset: [number, number, number];
};

export type OccupancyGrid = {
  origin: [number, number, number];
  voxelSize: number;
  dims: [number, number, number];
  minProb: number;
  cells: OccupancyCell[];
  collision: {
    risk: number;
    nearestDistanceM: number;
    wireRisk: number;
    solidRisk: number;
  } | null;
};

const text = new TextDecoder();

export function float16ToNumber(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export function numberToFloat16(value: number): number {
  const sample = new Float32Array(1);
  const raw = new Uint32Array(sample.buffer);
  sample[0] = value;
  const bits = raw[0];
  const sign = (bits >> 16) & 0x8000;
  const exponent = ((bits >> 23) & 0xff) - 127 + 15;
  const fraction = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    const denorm = (fraction | 0x800000) >> (1 - exponent);
    return sign | (denorm >> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | (fraction >> 13);
}

export function decodeGaussians(buffer: ArrayBuffer): GaussianCloud {
  const view = new DataView(buffer);
  const magic = text.decode(new Uint8Array(buffer, 0, 4));
  if (magic !== "AGS1") throw new Error(`bad gaussian magic ${magic}`);
  const version = view.getUint16(4, true);
  if (version !== 1) throw new Error(`unsupported gaussian version ${version}`);
  const batch = view.getUint32(8, true);
  const count = view.getUint32(12, true);
  const channels = view.getUint16(16, true);
  const shCoeffs = view.getUint16(18, true);
  if (channels !== CHANNELS || shCoeffs !== SH_COEFFS) {
    throw new Error(`unexpected layout ${channels} x ${shCoeffs}`);
  }
  const timestamp = view.getFloat32(20, true);
  const dt = view.getFloat32(24, true);
  const headerBytes = view.getUint32(28, true);
  const sourceSun: [number, number, number] = [
    view.getFloat32(32, true),
    view.getFloat32(36, true),
    view.getFloat32(40, true),
  ];
  const body = headerBytes;
  const packedFloats = batch * count * channels;
  const validBytes = batch * count;
  const expected = body + packedFloats * 2 + validBytes + validBytes * 2;
  if (buffer.byteLength !== expected) {
    throw new Error(`gaussian body length ${buffer.byteLength}, expected ${expected}`);
  }
  const n = batch * count;
  const xyz = new Float32Array(n * 3);
  const scale = new Float32Array(n * 3);
  const quat = new Float32Array(n * 4);
  const opacity = new Float32Array(n);
  const sh = new Float32Array(n * SH_COEFFS * 3);
  const velocity = new Float32Array(n * 3);
  const valid = new Uint8Array(n);
  const wireProb = new Float32Array(n);
  let cursor = body;
  for (let i = 0; i < n; i += 1) {
    const base = cursor + i * channels * 2;
    read3(view, base, xyz, i * 3);
    read3(view, base + 3 * 2, scale, i * 3);
    read4(view, base + 6 * 2, quat, i * 4);
    opacity[i] = readF16(view, base + 10 * 2);
    for (let k = 0; k < 48; k += 1) {
      sh[i * 48 + k] = readF16(view, base + (11 + k) * 2);
    }
    read3(view, base + 59 * 2, velocity, i * 3);
  }
  cursor += packedFloats * 2;
  valid.set(new Uint8Array(buffer, cursor, n));
  cursor += n;
  for (let i = 0; i < n; i += 1) wireProb[i] = readF16(view, cursor + i * 2);
  return {
    batch,
    count: n,
    timestamp,
    dt,
    sourceSun,
    xyz,
    scale,
    quat,
    opacity,
    sh,
    velocity,
    valid,
    wireProb,
  };
}

export function decodeOccupancy(buffer: ArrayBuffer): OccupancyGrid {
  const view = new DataView(buffer);
  const magic = text.decode(new Uint8Array(buffer, 0, 4));
  if (magic !== "AOC1") throw new Error(`bad occupancy magic ${magic}`);
  const version = view.getUint16(4, true);
  if (version !== 1) throw new Error(`unsupported occupancy version ${version}`);
  const flags = view.getUint16(6, true);
  const origin: [number, number, number] = [
    view.getFloat32(8, true),
    view.getFloat32(12, true),
    view.getFloat32(16, true),
  ];
  const voxelSize = view.getFloat32(20, true);
  const dims: [number, number, number] = [
    view.getUint32(24, true),
    view.getUint32(28, true),
    view.getUint32(32, true),
  ];
  const cellCount = view.getUint32(36, true);
  const risk = view.getFloat32(40, true);
  const nearest = view.getFloat32(44, true);
  const wireRisk = view.getFloat32(48, true);
  const solidRisk = view.getFloat32(52, true);
  const minProb = view.getFloat32(56, true);
  const cells: OccupancyCell[] = [];
  let cursor = OCCUPANCY_HEADER_BYTES;
  for (let index = 0; index < cellCount; index += 1) {
    cells.push({
      i: view.getUint16(cursor, true),
      j: view.getUint16(cursor + 2, true),
      k: view.getUint16(cursor + 4, true),
      occupancy: readF16(view, cursor + 6),
      wire: readF16(view, cursor + 8),
      direction: [readF16(view, cursor + 10), readF16(view, cursor + 12), readF16(view, cursor + 14)],
      offset: [readF16(view, cursor + 16), readF16(view, cursor + 18), readF16(view, cursor + 20)],
    });
    cursor += 24;
  }
  return {
    origin,
    voxelSize,
    dims,
    minProb,
    cells,
    collision:
      flags & 1
        ? {
            risk,
            nearestDistanceM: Number.isFinite(nearest) ? nearest : Number.POSITIVE_INFINITY,
            wireRisk,
            solidRisk,
          }
        : null,
  };
}

export function interpolatePosition(
  xyz: Float32Array,
  velocity: Float32Array,
  index: number,
  alpha: number,
  dt: number,
  out: { x: number; y: number; z: number },
): void {
  const step = alpha * dt;
  const p = index * 3;
  out.x = xyz[p] + velocity[p] * step;
  out.y = xyz[p + 1] + velocity[p + 1] * step;
  out.z = xyz[p + 2] + velocity[p + 2] * step;
}

function readF16(view: DataView, byte: number): number {
  return float16ToNumber(view.getUint16(byte, true));
}

function read3(view: DataView, byte: number, target: Float32Array, offset: number) {
  target[offset] = readF16(view, byte);
  target[offset + 1] = readF16(view, byte + 2);
  target[offset + 2] = readF16(view, byte + 4);
}

function read4(view: DataView, byte: number, target: Float32Array, offset: number) {
  target[offset] = readF16(view, byte);
  target[offset + 1] = readF16(view, byte + 2);
  target[offset + 2] = readF16(view, byte + 4);
  target[offset + 3] = readF16(view, byte + 6);
}

export function b64ToBuffer(value: string): ArrayBuffer {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}
