import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

import type { GaussianCloud } from "../codec";
import { SPLAT_FRAGMENT, SPLAT_VERTEX } from "../shaders";
import { relitMatrix, type Vec3 } from "../sh";
import { orderFarToNear } from "../sort";

const STRIDE = 16;
const TEX_WIDTH = 1024;
const DRAWING = new THREE.Vector2();

type Resources = {
  geometry: THREE.InstancedBufferGeometry;
  material: THREE.RawShaderMaterial;
  texture: THREE.DataTexture;
  depths: Float32Array;
  order: Uint32Array;
};

export function SplatField({
  cloud,
  alpha,
  dt,
  targetSun,
}: {
  cloud: GaussianCloud;
  alpha: number;
  dt: number;
  targetSun: Vec3;
}) {
  const resources = useMemo(() => buildResources(cloud), [cloud]);
  useEffect(() => {
    return () => {
      resources.geometry.dispose();
      resources.material.dispose();
      resources.texture.dispose();
    };
  }, [resources]);

  useFrame(({ camera, gl }) => {
    const { geometry, material, depths, order } = resources;
    const uniforms = material.uniforms;
    uniforms.uAlpha.value = alpha;
    uniforms.uDt.value = dt;
    uniforms.uView.value.copy(camera.matrixWorldInverse);
    uniforms.uProjection.value.copy(camera.projectionMatrix);
    uniforms.uCamera.value.copy(camera.position);
    const drawing = gl.getDrawingBufferSize(DRAWING);
    uniforms.uViewport.value.set(drawing.x, drawing.y);
    const fy = 0.5 * drawing.y * camera.projectionMatrix.elements[5];
    const fx = 0.5 * drawing.x * camera.projectionMatrix.elements[0];
    uniforms.uFocal.value.set(Math.abs(fx), Math.abs(fy));
    const relit = relitMatrix(cloud.sourceSun, targetSun);
    uniforms.uRelit.value.set(
      relit[0][0],
      relit[0][1],
      relit[0][2],
      relit[1][0],
      relit[1][1],
      relit[1][2],
      relit[2][0],
      relit[2][1],
      relit[2][2],
    );
    const view = camera.matrixWorldInverse.elements;
    const step = alpha * dt;
    for (let i = 0; i < cloud.count; i += 1) {
      const p = i * 3;
      const x = cloud.xyz[p] + cloud.velocity[p] * step;
      const y = cloud.xyz[p + 1] + cloud.velocity[p + 1] * step;
      const z = cloud.xyz[p + 2] + cloud.velocity[p + 2] * step;
      depths[i] = -(view[2] * x + view[6] * y + view[10] * z + view[14]);
    }
    orderFarToNear(depths, order);
    const attribute = geometry.getAttribute("splatIndex") as THREE.InstancedBufferAttribute;
    const array = attribute.array as Float32Array;
    for (let i = 0; i < order.length; i += 1) array[i] = order[i];
    attribute.needsUpdate = true;
  });

  return (
    <mesh geometry={resources.geometry} material={resources.material} frustumCulled={false} renderOrder={1} />
  );
}

function buildResources(cloud: GaussianCloud): Resources {
  const height = Math.max(1, Math.ceil((cloud.count * STRIDE) / TEX_WIDTH));
  const data = new Float32Array(TEX_WIDTH * height * 4);
  for (let i = 0; i < cloud.count; i += 1) {
    const texel = i * STRIDE;
    write4(data, TEX_WIDTH, texel, cloud.xyz[i * 3], cloud.xyz[i * 3 + 1], cloud.xyz[i * 3 + 2], cloud.opacity[i]);
    write4(
      data,
      TEX_WIDTH,
      texel + 1,
      cloud.scale[i * 3],
      cloud.scale[i * 3 + 1],
      cloud.scale[i * 3 + 2],
      cloud.valid[i] ? 1 : 0,
    );
    write4(data, TEX_WIDTH, texel + 2, cloud.quat[i * 4], cloud.quat[i * 4 + 1], cloud.quat[i * 4 + 2], cloud.quat[i * 4 + 3]);
    write4(
      data,
      TEX_WIDTH,
      texel + 3,
      cloud.velocity[i * 3],
      cloud.velocity[i * 3 + 1],
      cloud.velocity[i * 3 + 2],
      cloud.wireProb[i],
    );
    for (let k = 0; k < 48; k += 1) {
      const element = texel + 4 + Math.floor(k / 4);
      const x = element % TEX_WIDTH;
      const y = Math.floor(element / TEX_WIDTH);
      data[(y * TEX_WIDTH + x) * 4 + (k % 4)] = cloud.sh[i * 48 + k];
    }
  }
  const texture = new THREE.DataTexture(data, TEX_WIDTH, height, THREE.RGBAFormat, THREE.FloatType);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;

  const corners = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);
  const indices = new Float32Array(cloud.count);
  for (let i = 0; i < cloud.count; i += 1) indices[i] = i;
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute("corner", new THREE.BufferAttribute(corners, 2));
  geometry.setAttribute("splatIndex", new THREE.InstancedBufferAttribute(indices, 1));
  geometry.instanceCount = cloud.count;

  const material = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: SPLAT_VERTEX.replace("#version 300 es\n", ""),
    fragmentShader: SPLAT_FRAGMENT.replace("#version 300 es\n", ""),
    uniforms: {
      uData: { value: texture },
      uTexSize: { value: new THREE.Vector2(TEX_WIDTH, height) },
      uStride: { value: STRIDE },
      uAlpha: { value: 0 },
      uDt: { value: cloud.dt },
      uRelit: { value: new THREE.Matrix3() },
      uView: { value: new THREE.Matrix4() },
      uProjection: { value: new THREE.Matrix4() },
      uCamera: { value: new THREE.Vector3() },
      uFocal: { value: new THREE.Vector2(500, 500) },
      uViewport: { value: new THREE.Vector2(1, 1) },
    },
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.CustomBlending,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneMinusSrcAlphaFactor,
    blendSrcAlpha: THREE.OneFactor,
    blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
    toneMapped: false,
  });

  return {
    geometry,
    material,
    texture,
    depths: new Float32Array(cloud.count),
    order: new Uint32Array(cloud.count),
  };
}

function write4(data: Float32Array, width: number, texel: number, a: number, b: number, c: number, d: number) {
  const x = texel % width;
  const y = Math.floor(texel / width);
  const offset = (y * width + x) * 4;
  data[offset] = a;
  data[offset + 1] = b;
  data[offset + 2] = c;
  data[offset + 3] = d;
}
