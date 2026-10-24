import { describe, expect, it } from "vitest";

import { decodeGaussians, decodeOccupancy, interpolatePosition, numberToFloat16 } from "./codec";
import { SH_C0 } from "./sh";
import { SPLAT_VERTEX } from "./shaders";

describe("AGS1", () => {
  it("reads a one-gaussian float16 stream", () => {
    const count = 1;
    const channels = 62;
    const header = 64;
    const body = channels * 2 + 1 + 2;
    const buffer = new ArrayBuffer(header + body);
    const view = new DataView(buffer);
    view.setUint8(0, "A".charCodeAt(0));
    view.setUint8(1, "G".charCodeAt(0));
    view.setUint8(2, "S".charCodeAt(0));
    view.setUint8(3, "1".charCodeAt(0));
    view.setUint16(4, 1, true);
    view.setUint16(6, 1, true);
    view.setUint32(8, 1, true);
    view.setUint32(12, count, true);
    view.setUint16(16, channels, true);
    view.setUint16(18, 16, true);
    view.setFloat32(20, 0.4, true);
    view.setFloat32(24, 0.1, true);
    view.setUint32(28, header, true);
    view.setFloat32(32, 0, true);
    view.setFloat32(36, 0, true);
    view.setFloat32(40, 1, true);
    writeF16(view, header, 1.5);
    writeF16(view, header + 2, -2);
    writeF16(view, header + 4, 0.25);
    writeF16(view, header + 10 * 2, 0.8);
    writeF16(view, header + 59 * 2, 1.25);
    view.setUint8(header + channels * 2, 1);
    writeF16(view, header + channels * 2 + 1, 0.5);
    const cloud = decodeGaussians(buffer);
    expect(cloud.count).toBe(1);
    expect(cloud.timestamp).toBeCloseTo(0.4);
    expect(cloud.dt).toBeCloseTo(0.1);
    expect(cloud.sourceSun[2]).toBeCloseTo(1);
    expect(cloud.xyz[0]).toBeCloseTo(1.5, 2);
    expect(cloud.xyz[1]).toBeCloseTo(-2, 2);
    expect(cloud.opacity[0]).toBeCloseTo(0.8, 2);
    expect(cloud.velocity[0]).toBeCloseTo(1.25, 2);
    expect(cloud.valid[0]).toBe(1);
    expect(cloud.wireProb[0]).toBeCloseTo(0.5, 2);
  });

  it("interpolates position with velocity", () => {
    const xyz = new Float32Array([1, 0, 0]);
    const velocity = new Float32Array([2, 0, 0]);
    const out = { x: 0, y: 0, z: 0 };
    interpolatePosition(xyz, velocity, 0, 0.5, 0.2, out);
    expect(out.x).toBeCloseTo(1.2);
  });
});

describe("AOC1", () => {
  it("reads one sparse cell and a collision flag", () => {
    const buffer = new ArrayBuffer(64 + 24);
    const view = new DataView(buffer);
    view.setUint8(0, "A".charCodeAt(0));
    view.setUint8(1, "O".charCodeAt(0));
    view.setUint8(2, "C".charCodeAt(0));
    view.setUint8(3, "1".charCodeAt(0));
    view.setUint16(4, 1, true);
    view.setUint16(6, 1, true);
    view.setFloat32(20, 0.25, true);
    view.setUint32(24, 4, true);
    view.setUint32(28, 4, true);
    view.setUint32(32, 4, true);
    view.setUint32(36, 1, true);
    view.setFloat32(40, 0.9, true);
    view.setFloat32(44, 0.1, true);
    view.setFloat32(48, 0.9, true);
    view.setFloat32(52, 0.2, true);
    view.setFloat32(56, 0.5, true);
    view.setUint16(64, 1, true);
    view.setUint16(66, 2, true);
    view.setUint16(68, 3, true);
    writeF16(view, 70, 0.2);
    writeF16(view, 72, 0.95);
    const grid = decodeOccupancy(buffer);
    expect(grid.dims).toEqual([4, 4, 4]);
    expect(grid.cells).toHaveLength(1);
    expect(grid.cells[0].i).toBe(1);
    expect(grid.cells[0].wire).toBeCloseTo(0.95, 2);
    expect(grid.collision?.wireRisk).toBeCloseTo(0.9);
  });
});

function writeF16(view: DataView, byte: number, value: number) {
  view.setUint16(byte, numberToFloat16(value), true);
}

describe("shader contract", () => {
  it("uses the rasterizer SH constant", () => {
    expect(SPLAT_VERTEX).toContain(String(SH_C0));
    expect(SPLAT_VERTEX).toContain("uRelit");
    expect(SPLAT_VERTEX).toContain("uAlpha");
  });
});
