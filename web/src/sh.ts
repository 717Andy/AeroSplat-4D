/**
 * Sun-slider math shared with `aerosplat.models.relight`.
 *
 * `rotationBetween(source, target)` is the matrix R such that R maps `source`
 * onto `target`. `evaluate_relit_sh` then evaluates SH at R^T * view_dir.
 * The shader receives that transpose as `uRelit`.
 *
 * Degree-3 evaluation matches `aerosplat.models.rasterizer.eval_sh`, including
 * the rasterizer's +0.5. Keep the constants identical to `shaders.ts`.
 */

export const SH_C0 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const C2 = [
  1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
  0.5462742152960396,
];
const C3 = [
  -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
  -0.4570457994644658, 1.445305721320277, -0.5900435899266435,
];

export type Vec3 = [number, number, number];
export type Mat3 = [Vec3, Vec3, Vec3];

export function normalize(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (n < 1e-8) throw new Error("zero vector");
  return [v[0] / n, v[1] / n, v[2] / n];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function sunFromAzEl(azimuthDeg: number, elevationDeg: number): Vec3 {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  return normalize([Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)]);
}

export function azElFromSun(sun: Vec3): { azimuth: number; elevation: number } {
  const u = normalize(sun);
  const elevation = (Math.asin(Math.max(-1, Math.min(1, u[2]))) * 180) / Math.PI;
  const azimuth = (Math.atan2(u[1], u[0]) * 180) / Math.PI;
  return { azimuth: (azimuth + 360) % 360, elevation };
}

/** Rows of R. R @ source = target. Mirrors `rotation_between` in relight.py. */
export function rotationBetween(source: Vec3, target: Vec3): Mat3 {
  const src = normalize(source);
  const dst = normalize(target);
  const c = cross(src, dst);
  const d = Math.max(-1, Math.min(1, dot(src, dst)));
  const sin = Math.max(1e-8, Math.hypot(c[0], c[1], c[2]));
  const axis: Vec3 = [c[0] / sin, c[1] / sin, c[2] / sin];
  const angle = Math.acos(d);
  const eye: Mat3 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  if (d > 0.9995) return eye;
  if (d < -0.9995) {
    const ortho = orthogonal(src);
    return outerHalfTurn(ortho);
  }
  const k = skew(axis);
  const kk = mul(k, k);
  const s = Math.sin(angle);
  const co = 1 - Math.cos(angle);
  const rot = eye.map((row, r) =>
    row.map((value, col) => value + s * k[r][col] + co * kk[r][col]),
  ) as Mat3;
  return rot;
}

/** Matrix passed to the shader: columns of R^T, i.e. `uRelit * dir = R^T dir`. */
export function relitMatrix(source: Vec3, target: Vec3): Mat3 {
  const rotation = rotationBetween(source, target);
  return [
    [rotation[0][0], rotation[1][0], rotation[2][0]],
    [rotation[0][1], rotation[1][1], rotation[2][1]],
    [rotation[0][2], rotation[1][2], rotation[2][2]],
  ];
}

export function mulVec(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

/** `sh` is 16*3, coefficient-major RGB. Includes the rasterizer +0.5. */
export function evalSh(sh: ArrayLike<number>, dirIn: Vec3): Vec3 {
  const dir = normalize(dirIn);
  const x = dir[0];
  const y = dir[1];
  const z = dir[2];
  const c = (coeff: number, channel: number) => sh[coeff * 3 + channel];
  const result: Vec3 = [0, 0, 0];
  for (let channel = 0; channel < 3; channel += 1) {
    let value = SH_C0 * c(0, channel);
    value += -SH_C1 * y * c(1, channel) + SH_C1 * z * c(2, channel) - SH_C1 * x * c(3, channel);
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    value +=
      C2[0] * x * y * c(4, channel) +
      C2[1] * y * z * c(5, channel) +
      C2[2] * (2 * zz - xx - yy) * c(6, channel) +
      C2[3] * x * z * c(7, channel) +
      C2[4] * (xx - yy) * c(8, channel);
    value +=
      C3[0] * y * (3 * xx - yy) * c(9, channel) +
      C3[1] * x * y * z * c(10, channel) +
      C3[2] * y * (4 * zz - xx - yy) * c(11, channel) +
      C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * c(12, channel) +
      C3[4] * x * (4 * zz - xx - yy) * c(13, channel) +
      C3[5] * z * (xx - yy) * c(14, channel) +
      C3[6] * x * (xx - 3 * yy) * c(15, channel);
    result[channel] = value + 0.5;
  }
  return result;
}

function skew(axis: Vec3): Mat3 {
  const [x, y, z] = axis;
  return [
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0],
  ];
}

function mul(a: Mat3, b: Mat3): Mat3 {
  const out: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r += 1) {
    for (let col = 0; col < 3; col += 1) {
      out[r][col] = a[r][0] * b[0][col] + a[r][1] * b[1][col] + a[r][2] * b[2][col];
    }
  }
  return out;
}

function orthogonal(vector: Vec3): Vec3 {
  const other: Vec3 = Math.abs(vector[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
  return normalize(cross(vector, other));
}

function outerHalfTurn(axis: Vec3): Mat3 {
  const o = outer(axis, axis);
  return [
    [2 * o[0][0] - 1, 2 * o[0][1], 2 * o[0][2]],
    [2 * o[1][0], 2 * o[1][1] - 1, 2 * o[1][2]],
    [2 * o[2][0], 2 * o[2][1], 2 * o[2][2] - 1],
  ];
}

function outer(a: Vec3, b: Vec3): Mat3 {
  return [
    [a[0] * b[0], a[0] * b[1], a[0] * b[2]],
    [a[1] * b[0], a[1] * b[1], a[1] * b[2]],
    [a[2] * b[0], a[2] * b[1], a[2] * b[2]],
  ];
}
