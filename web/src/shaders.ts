/** Keep SH constants identical to `sh.ts` / `aerosplat.models.rasterizer.eval_sh`. */

export const SPLAT_VERTEX = /* glsl */ `#version 300 es
precision highp float;

in vec2 corner;
in float splatIndex;

uniform sampler2D uData;
uniform vec2 uTexSize;
uniform float uStride;
uniform float uAlpha;
uniform float uDt;
uniform mat3 uRelit;
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uCamera;
uniform vec2 uFocal;
uniform vec2 uViewport;

out vec3 vColor;
out vec2 vCenter;
out vec3 vConic;
out float vOpacity;

const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;

vec4 fetchTexel(float index) {
  float x = mod(index, uTexSize.x);
  float y = floor(index / uTexSize.x);
  return texelFetch(uData, ivec2(int(x), int(y)), 0);
}

float shElement(vec4 band[12], int element) {
  int texel = element / 4;
  int comp = element - texel * 4;
  return band[texel][comp];
}

vec3 shCoeff(vec4 band[12], int index) {
  int element = index * 3;
  return vec3(shElement(band, element), shElement(band, element + 1), shElement(band, element + 2));
}

vec3 evalSh(vec4 band[12], vec3 dir) {
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;
  vec3 c0 = shCoeff(band, 0);
  vec3 c1 = shCoeff(band, 1);
  vec3 c2 = shCoeff(band, 2);
  vec3 c3 = shCoeff(band, 3);
  vec3 c4 = shCoeff(band, 4);
  vec3 c5 = shCoeff(band, 5);
  vec3 c6 = shCoeff(band, 6);
  vec3 c7 = shCoeff(band, 7);
  vec3 c8 = shCoeff(band, 8);
  vec3 c9 = shCoeff(band, 9);
  vec3 c10 = shCoeff(band, 10);
  vec3 c11 = shCoeff(band, 11);
  vec3 c12 = shCoeff(band, 12);
  vec3 c13 = shCoeff(band, 13);
  vec3 c14 = shCoeff(band, 14);
  vec3 c15 = shCoeff(band, 15);
  vec3 result = SH_C0 * c0;
  result += -SH_C1 * y * c1 + SH_C1 * z * c2 - SH_C1 * x * c3;
  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float xy = x * y;
  float yz = y * z;
  float xz = x * z;
  result += 1.0925484305920792 * xy * c4;
  result += -1.0925484305920792 * yz * c5;
  result += 0.31539156525252005 * (2.0 * zz - xx - yy) * c6;
  result += -1.0925484305920792 * xz * c7;
  result += 0.5462742152960396 * (xx - yy) * c8;
  result += -0.5900435899266435 * y * (3.0 * xx - yy) * c9;
  result += 2.890611442640554 * xy * z * c10;
  result += -0.4570457994644658 * y * (4.0 * zz - xx - yy) * c11;
  result += 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * c12;
  result += -0.4570457994644658 * x * (4.0 * zz - xx - yy) * c13;
  result += 1.445305721320277 * z * (xx - yy) * c14;
  result += -0.5900435899266435 * x * (xx - 3.0 * yy) * c15;
  return result;
}

void main() {
  float base = splatIndex * uStride;
  vec4 p0 = fetchTexel(base);
  vec4 p1 = fetchTexel(base + 1.0);
  vec4 p2 = fetchTexel(base + 2.0);
  vec4 p3 = fetchTexel(base + 3.0);
  vec4 band[12];
  for (int i = 0; i < 12; i++) {
    band[i] = fetchTexel(base + 4.0 + float(i));
  }
  vec3 world = p0.xyz + p3.xyz * (uAlpha * uDt);
  float opacity = p0.w;
  vec3 scale = max(p1.xyz, vec3(1e-4));
  float valid = p1.w;
  vec4 q = p2;
  vec3 cam = (uView * vec4(world, 1.0)).xyz;
  float depth = -cam.z;
  if (valid < 0.5 || opacity < 1.0 / 255.0 || depth < 0.05) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vColor = vec3(0.0);
    vCenter = vec2(0.0);
    vConic = vec3(1.0, 0.0, 1.0);
    vOpacity = 0.0;
    return;
  }

  float w = q.x;
  float x = q.y;
  float y = q.z;
  float zq = q.w;
  vec3 col0 = vec3(1.0 - 2.0 * (y * y + zq * zq), 2.0 * (x * y + zq * w), 2.0 * (x * zq - y * w));
  vec3 col1 = vec3(2.0 * (x * y - zq * w), 1.0 - 2.0 * (x * x + zq * zq), 2.0 * (y * zq + x * w));
  vec3 col2 = vec3(2.0 * (x * zq + y * w), 2.0 * (y * zq - x * w), 1.0 - 2.0 * (x * x + y * y));
  mat3 rot = mat3(col0, col1, col2);
  vec3 s2 = scale * scale;
  mat3 sigma = rot * mat3(s2.x, 0.0, 0.0, 0.0, s2.y, 0.0, 0.0, 0.0, s2.z) * transpose(rot);
  mat3 viewRot = mat3(uView);
  mat3 covCam = viewRot * sigma * transpose(viewRot);

  float a00 = covCam[0][0];
  float a01 = -covCam[1][0];
  float a02 = -covCam[2][0];
  float a11 = covCam[1][1];
  float a12 = covCam[2][1];
  float a22 = covCam[2][2];
  float z = depth;
  float cx = cam.x;
  float cy = -cam.y;
  float invZ = 1.0 / z;
  float jxx = uFocal.x * invZ;
  float jyy = uFocal.y * invZ;
  float jxz = -uFocal.x * cx * invZ * invZ;
  float jyz = -uFocal.y * cy * invZ * invZ;
  float m0x = a00 * jxx + a02 * jxz;
  float m0y = a01 * jxx + a12 * jxz;
  float m0z = a02 * jxx + a22 * jxz;
  float cov00 = jxx * m0x + jxz * m0z;
  float m1x = a01 * jyy + a02 * jyz;
  float m1y = a11 * jyy + a12 * jyz;
  float m1z = a12 * jyy + a22 * jyz;
  float cov11 = jyy * m1y + jyz * m1z;
  float cov01 = jxx * m1x + jxz * m1z;
  cov00 += 0.3;
  cov11 += 0.3;
  float det = cov00 * cov11 - cov01 * cov01;

  vec4 clip = uProjection * vec4(cam, 1.0);
  vec2 ndcCenter = clip.xy / max(clip.w, 1e-6);
  vec2 center = (ndcCenter * 0.5 + 0.5) * uViewport;
  vec2 axis1;
  vec2 axis2;
  float radius1;
  float radius2;
  vec3 conic;
  if (det > 1e-4) {
    float mid = 0.5 * (cov00 + cov11);
    float disc = sqrt(max(mid * mid - det, 0.0));
    float lambda1 = max(mid + disc, 1e-6);
    float lambda2 = max(mid - disc, 1e-6);
    radius1 = min(sqrt(lambda1) * 3.0, 384.0);
    radius2 = min(sqrt(lambda2) * 3.0, 384.0);
    axis1 = abs(cov01) > 1e-4 ? normalize(vec2(cov01, lambda1 - cov00)) : vec2(1.0, 0.0);
    axis2 = vec2(-axis1.y, axis1.x);
    conic = vec3(cov11 / det, -cov01 / det, cov00 / det);
  } else {
    float radius = clamp(max(scale.x, max(scale.y, scale.z)) * jxx, 1.5, 48.0);
    radius1 = radius;
    radius2 = radius;
    axis1 = vec2(1.0, 0.0);
    axis2 = vec2(0.0, 1.0);
    float inv = 1.0 / max((radius / 3.0) * (radius / 3.0), 1e-4);
    conic = vec3(inv, 0.0, inv);
  }
  vec2 offsetCv = corner.x * axis1 * radius1 + corner.y * axis2 * radius2;
  vec2 pixel = center + vec2(offsetCv.x, -offsetCv.y);
  vec2 ndc = pixel / uViewport * 2.0 - 1.0;
  gl_Position = vec4(ndc, clip.z / clip.w, 1.0);
  vec3 viewDir = normalize(uRelit * normalize(world - uCamera));
  vColor = clamp(evalSh(band, viewDir) + 0.5, 0.0, 1.0);
  vCenter = center;
  vConic = conic;
  vOpacity = opacity;
}
`;

export const SPLAT_FRAGMENT = /* glsl */ `#version 300 es
precision highp float;

in vec3 vColor;
in vec2 vCenter;
in vec3 vConic;
in float vOpacity;
out vec4 fragColor;

void main() {
  vec2 delta = gl_FragCoord.xy - vCenter;
  delta.y = -delta.y;
  float power = -0.5 * (vConic.x * delta.x * delta.x + vConic.z * delta.y * delta.y) - vConic.y * delta.x * delta.y;
  if (power > 0.0) discard;
  float alpha = vOpacity * exp(power);
  if (alpha < 1.0 / 255.0) discard;
  fragColor = vec4(vColor * alpha, alpha);
}
`;
