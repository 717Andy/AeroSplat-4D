/** Counting sort so a million splats stay sortable without an n log n compare. */

export function orderFarToNear(depths: ArrayLike<number>, out: Uint32Array): void {
  const n = out.length;
  if (n === 0) return;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const depth = depths[i];
    if (depth < min) min = depth;
    if (depth > max) max = depth;
  }
  const bins = 4096;
  const counts = new Uint32Array(bins);
  const keys = new Uint32Array(n);
  const span = max - min;
  const scale = span > 1e-8 ? (bins - 1) / span : 0;
  for (let i = 0; i < n; i += 1) {
    const quantized = Math.round((depths[i] - min) * scale);
    const bin = bins - 1 - Math.max(0, Math.min(bins - 1, quantized));
    keys[i] = bin;
    counts[bin] += 1;
  }
  let sum = 0;
  for (let i = 0; i < bins; i += 1) {
    const count = counts[i];
    counts[i] = sum;
    sum += count;
  }
  for (let i = 0; i < n; i += 1) {
    const bin = keys[i];
    out[counts[bin]] = i;
    counts[bin] += 1;
  }
}
