import type { Vec3 } from "./sh";

export type Mat3x3 = number[][];

/** World position of the camera center. ``X_cam = R X_world + t``. */
export function cameraCenter(rotation: Mat3x3, translation: number[]): Vec3 {
  const r0 = rotation[0];
  const r1 = rotation[1];
  const r2 = rotation[2];
  const t = translation;
  return [
    -(r0[0] * t[0] + r1[0] * t[1] + r2[0] * t[2]),
    -(r0[1] * t[0] + r1[1] * t[1] + r2[1] * t[2]),
    -(r0[2] * t[0] + r1[2] * t[1] + r2[2] * t[2]),
  ];
}

/** Camera-frame point to the flight frame. */
export function cameraToWorld(rotation: Mat3x3, translation: number[], point: Vec3): Vec3 {
  const delta = [point[0] - translation[0], point[1] - translation[1], point[2] - translation[2]];
  return [
    rotation[0][0] * delta[0] + rotation[1][0] * delta[1] + rotation[2][0] * delta[2],
    rotation[0][1] * delta[0] + rotation[1][1] * delta[1] + rotation[2][1] * delta[2],
    rotation[0][2] * delta[0] + rotation[1][2] * delta[1] + rotation[2][2] * delta[2],
  ];
}

export type RigCamera = {
  name: string;
  width: number;
  height: number;
  K: number[][];
  R: number[][];
  t: number[];
  imageUrl: string;
};

export function frustumCorners(camera: RigCamera, depth: number): Vec3[] {
  const fx = camera.K[0][0];
  const fy = camera.K[1][1];
  const cx = camera.K[0][2];
  const cy = camera.K[1][2];
  const corners: Array<[number, number]> = [
    [0, 0],
    [camera.width - 1, 0],
    [camera.width - 1, camera.height - 1],
    [0, camera.height - 1],
  ];
  return corners.map(([u, v]) => {
    const cam: Vec3 = [((u - cx) * depth) / fx, ((v - cy) * depth) / fy, depth];
    return cameraToWorld(camera.R, camera.t, cam);
  });
}
