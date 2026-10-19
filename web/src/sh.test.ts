import { describe, expect, it } from "vitest";

import { cameraCenter, cameraToWorld } from "./rig";
import { azElFromSun, evalSh, mulVec, relitMatrix, rotationBetween, sunFromAzEl, type Vec3 } from "./sh";
import { orderFarToNear } from "./sort";

const eye = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

describe("sun rotation", () => {
  it("maps +Z onto +X", () => {
    const source: Vec3 = [0, 0, 1];
    const target: Vec3 = [1, 0, 0];
    const mapped = mulVec(rotationBetween(source, target), source);
    expect(mapped[0]).toBeCloseTo(1, 5);
    expect(mapped[1]).toBeCloseTo(0, 5);
    expect(mapped[2]).toBeCloseTo(0, 5);
    const back = mulVec(relitMatrix(source, target), target);
    expect(back[0]).toBeCloseTo(0, 5);
    expect(back[2]).toBeCloseTo(1, 5);
  });

  it("keeps a degree-0 color view independent", () => {
    const rgb: Vec3 = [0.2, 0.4, 0.6];
    const sh = new Float32Array(48);
    sh[0] = (rgb[0] - 0.5) / 0.28209479177387814;
    sh[1] = (rgb[1] - 0.5) / 0.28209479177387814;
    sh[2] = (rgb[2] - 0.5) / 0.28209479177387814;
    const color = evalSh(sh, [0.3, -0.2, 0.9]);
    expect(color[0]).toBeCloseTo(0.2, 5);
    expect(color[1]).toBeCloseTo(0.4, 5);
    expect(color[2]).toBeCloseTo(0.6, 5);
  });

  it("round-trips azimuth and elevation", () => {
    const sun = sunFromAzEl(30, 50);
    const angles = azElFromSun(sun);
    expect(angles.azimuth).toBeCloseTo(30, 4);
    expect(angles.elevation).toBeCloseTo(50, 4);
  });
});

describe("rig", () => {
  it("puts the camera center at -R^T t", () => {
    const center = cameraCenter(eye, [0, 0, -5]);
    expect(center[0]).toBeCloseTo(0);
    expect(center[1]).toBeCloseTo(0);
    expect(center[2]).toBeCloseTo(5);
    const world = cameraToWorld(eye, [0, 0, 0], [0, 0, 2]);
    expect(world[2]).toBeCloseTo(2);
  });
});

describe("sort", () => {
  it("orders far splats first", () => {
    const depths = [1, 5, 3, 5];
    const order = new Uint32Array(depths.length);
    orderFarToNear(depths, order);
    expect(Array.from(order.slice(0, 2)).sort()).toEqual([1, 3]);
    expect(order[2]).toBe(2);
    expect(order[3]).toBe(0);
  });
});
