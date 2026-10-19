import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import { b64ToBuffer, decodeGaussians, decodeOccupancy, type GaussianCloud, type OccupancyGrid } from "./codec";
import type { RigCamera } from "./rig";
import { CameraRig, CorridorLine, OccupancyBoxes, VelocityArrows, WireLines } from "./scene/Overlays";
import { SplatField } from "./scene/SplatField";
import { azElFromSun, sunFromAzEl } from "./sh";

type Episode = {
  id: string;
  samples: number;
  title: string;
  description: string;
  source_sun: number[] | null;
  corridor: number[][] | null;
  corridor_radius_m: number | null;
};

type FrameCamera = {
  name: string;
  width: number;
  height: number;
  K: number[][];
  R: number[][];
  t: number[];
  image_url: string;
  glare_url: string;
  has_glare: boolean;
  sync: string;
  timestamp: number;
};

type Frame = {
  index: number;
  timestamp: number;
  dt: number;
  samples: number;
  cameras: FrameCamera[];
  cloud_url: string | null;
  corridor: number[][] | null;
  corridor_radius_m: number | null;
};

type Health = {
  gpu: { available: boolean; name: string; vram_total_bytes: number | null; vram_used_bytes: number | null };
  latency_ms: Record<string, number>;
  throughput_fps: number | null;
  model: { loaded: boolean; name: string; init: string | null; device: string | null };
};

type GlareMode = "clean" | "glare" | "composite";

const API = import.meta.env.VITE_API ?? "";

export function App() {
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [episodeId, setEpisodeId] = useState<string>("");
  const [frame, setFrame] = useState<Frame | null>(null);
  const [scrub, setScrub] = useState(0);
  const [cloud, setCloud] = useState<GaussianCloud | null>(null);
  const [grid, setGrid] = useState<OccupancyGrid | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string>("");
  const [showOcc, setShowOcc] = useState(true);
  const [showWires, setShowWires] = useState(true);
  const [showVel, setShowVel] = useState(true);
  const [showRig, setShowRig] = useState(true);
  const [radius, setRadius] = useState(0.6);
  const [azimuth, setAzimuth] = useState(25);
  const [elevation, setElevation] = useState(70);
  const [sunLocked, setSunLocked] = useState(true);
  const [glareMode, setGlareMode] = useState<GlareMode>("composite");
  const [cameraMode, setCameraMode] = useState<Record<string, GlareMode | "follow">>({});
  const [glareOverride, setGlareOverride] = useState<Record<string, string>>({});
  const [cloudLabel, setCloudLabel] = useState("recording");
  const [busy, setBusy] = useState(false);
  const sunLockedRef = useRef(true);
  sunLockedRef.current = sunLocked;

  const episode = episodes.find((item) => item.id === episodeId) ?? null;
  const index = Math.min(Math.floor(scrub), Math.max((frame?.samples ?? 1) - 1, 0));
  const alpha = frame && index < frame.samples - 1 ? scrub - index : 0;
  const dt = frame?.dt ?? cloud?.dt ?? 0;
  const targetSun = useMemo(() => sunFromAzEl(azimuth, elevation), [azimuth, elevation]);
  const corridor = useMemo(
    () => (frame?.corridor ?? episode?.corridor ?? []) as [number, number, number][],
    [frame, episode],
  );

  useEffect(() => {
    void getJSON<Episode[]>("/api/v1/episodes")
      .then((rows) => {
        setEpisodes(rows);
        if (rows[0]) setEpisodeId(rows[0].id);
      })
      .catch((reason: unknown) => setError(String(reason)));
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void getJSON<Health>("/api/v1/health")
        .then(setHealth)
        .catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!episodeId) return;
    let cancelled = false;
    setGlareOverride({});
    setCloudLabel("recording");
    void (async () => {
      try {
        const next = await getJSON<Frame>(`/api/v1/episodes/${episodeId}/frames/${index}`);
        if (cancelled) return;
        setFrame(next);
        if (next.cloud_url) {
          const bytes = await getBytes(next.cloud_url);
          if (cancelled) return;
          const decoded = decodeGaussians(bytes);
          setCloud(decoded);
          if (sunLockedRef.current) {
            const angles = azElFromSun(decoded.sourceSun);
            setAzimuth(angles.azimuth);
            setElevation(angles.elevation);
          }
        } else {
          setCloud(null);
        }
      } catch (reason) {
        if (!cancelled) setError(String(reason));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [episodeId, index]);

  useEffect(() => {
    if (!episodeId) return;
    const points = corridor.map((point) => point.join(",")).join(";");
    const query = new URLSearchParams({
      episode: episodeId,
      index: String(index),
      radius: String(radius),
    });
    if (points) query.set("corridor", points);
    const timer = window.setTimeout(() => {
      void getJSON<OccupancyResponse>(`/api/v1/occupancy?${query.toString()}`)
        .then((body) => setGrid(decodeOccupancy(b64ToBuffer(body.occupancy_b64))))
        .catch((reason: unknown) => setError(String(reason)));
    }, 120);
    return () => window.clearTimeout(timer);
  }, [episodeId, index, radius, corridor]);

  const rig = useMemo<RigCamera[]>(() => {
    if (!frame || !showRig) return [];
    return frame.cameras.map((camera) => ({
      name: camera.name,
      width: camera.width,
      height: camera.height,
      K: camera.K,
      R: camera.R,
      t: camera.t,
      imageUrl: `${API}${camera.image_url}`,
    }));
  }, [frame, showRig]);

  const predict = useCallback(async () => {
    if (!frame) return;
    setBusy(true);
    setError("");
    try {
      const cameras = await Promise.all(
        frame.cameras.map(async (camera) => ({
          name: camera.name,
          timestamp: camera.timestamp,
          K: camera.K,
          R: camera.R,
          t: camera.t,
          image_png_base64: await urlToB64(`${API}${camera.image_url}`),
        })),
      );
      const body = await postJSON<InferResponse>("/api/v1/infer", {
        format: "json",
        sun_world: targetSun,
        cameras,
      });
      const decoded = decodeGaussians(b64ToBuffer(body.gaussian_b64));
      setCloud(decoded);
      setGrid(decodeOccupancy(b64ToBuffer(body.occupancy_b64)));
      const overrides: Record<string, string> = {};
      for (const [name, layer] of Object.entries(body.glare)) {
        overrides[name] = `data:image/png;base64,${layer.png_b64}`;
      }
      setGlareOverride(overrides);
      setCloudLabel(body.model.init === "checkpoint" ? "checkpoint" : "random init");
      setSunLocked(false);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(false);
    }
  }, [frame, targetSun]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight") setScrub((value) => value + 0.02);
      if (event.key === "ArrowLeft") setScrub((value) => Math.max(0, value - 0.02));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const time = (frame?.timestamp ?? 0) + alpha * dt;
  const warned = (grid?.collision?.risk ?? 0) > 0.5;
  const maxScrub = Math.max((frame?.samples ?? 1) - 1, 0);

  return (
    <div className="app">
      <header className="top">
        <div className="mark">
          <span>UAM</span>
          <strong>AeroSplat</strong>
          <em>4D workbench</em>
        </div>
        <label className="episode">
          Episode
          <select value={episodeId} onChange={(event) => { setEpisodeId(event.target.value); setScrub(0); }}>
            {episodes.map((item) => (
              <option key={item.id} value={item.id}>
                {item.title}
              </option>
            ))}
          </select>
        </label>
        <HealthReadout health={health} />
      </header>
      <div className="body">
        <div className="viewport">
          <Canvas
            camera={{ position: [9.5, -11, 6.2], up: [0, 0, 1], fov: 42, near: 0.05, far: 200 }}
            gl={{ antialias: true, toneMapping: THREE.NoToneMapping }}
            onCreated={({ camera }) => {
              camera.up.set(0, 0, 1);
              camera.lookAt(3, 0, 1.4);
            }}
          >
            <color attach="background" args={["#0c1210"]} />
            <EnuAxes />
            <gridHelper args={[18, 18, "#2f6f66", "#1b2823"]} rotation={[Math.PI / 2, 0, 0]} position={[3, 0, 0]} />
            {cloud && cloud.count > 0 ? (
              <SplatField cloud={cloud} alpha={alpha} dt={dt} targetSun={targetSun} />
            ) : null}
            {showOcc && grid ? <OccupancyBoxes grid={grid} /> : null}
            {showWires && grid ? <WireLines grid={grid} /> : null}
            {showVel && cloud ? <VelocityArrows cloud={cloud} alpha={alpha} dt={dt} /> : null}
            {corridor.length > 1 ? <CorridorLine points={corridor} warned={warned} /> : null}
            {showRig ? <CameraRig cameras={rig} /> : null}
            <OrbitControls makeDefault target={[3, 0, 1.3]} enableDamping dampingFactor={0.12} />
          </Canvas>
          <div className="hud">
            <span>{cloud ? `${cloud.count.toLocaleString()} splats` : "no cloud"}</span>
            <span className="tag">{cloudLabel}</span>
            <span>t {time.toFixed(3)} s</span>
            <span>α {alpha.toFixed(2)}</span>
          </div>
          {warned && grid?.collision ? (
            <div className="alarm">
              Corridor risk {grid.collision.risk.toFixed(2)} · wire {grid.collision.wireRisk.toFixed(2)} · solid{" "}
              {grid.collision.solidRisk.toFixed(2)} · clearance{" "}
              {Number.isFinite(grid.collision.nearestDistanceM)
                ? `${grid.collision.nearestDistanceM.toFixed(2)} m`
                : "clear"}
            </div>
          ) : null}
          {error ? <div className="error">{error}</div> : null}
        </div>
        <aside className="rail">
          <h2>Cameras</h2>
          <p className="hint">ENU, Z up. OpenCV frustums. Glare is an image layer, not a Gaussian.</p>
          {frame?.cameras.map((camera) => {
            const choice = cameraMode[camera.name] ?? "follow";
            const mode: GlareMode = choice === "follow" ? glareMode : choice;
            const glareUrl = glareOverride[camera.name] ?? (camera.has_glare ? `${API}${camera.glare_url}` : "");
            return (
              <figure key={camera.name} className="cam">
                <figcaption>
                  <b>{camera.name}</b>
                  <span>
                    {camera.width}×{camera.height} · {camera.sync}
                  </span>
                  <select
                    aria-label={`${camera.name} glare`}
                    value={cameraMode[camera.name] ?? "follow"}
                    onChange={(event) =>
                      setCameraMode((current) => ({
                        ...current,
                        [camera.name]: event.target.value as GlareMode | "follow",
                      }))
                    }
                  >
                    <option value="follow">follow</option>
                    <option value="clean">clean</option>
                    <option value="glare">glare</option>
                    <option value="composite">composite</option>
                  </select>
                </figcaption>
                <CameraStill
                  imageUrl={`${API}${camera.image_url}`}
                  glareUrl={glareUrl}
                  mode={mode}
                />
              </figure>
            );
          })}
        </aside>
      </div>
      <footer className="deck">
        <label className="scrub">
          <span>
            Timeline <b>{index}</b>
            <i> / {maxScrub}</i>
          </span>
          <input
            type="range"
            min={0}
            max={maxScrub}
            step={0.001}
            value={Math.min(scrub, maxScrub)}
            onChange={(event) => setScrub(Number(event.target.value))}
            aria-label="Timeline between frame t and t+1"
          />
        </label>
        <div className="toggles">
          <Toggle label="Occupancy" on={showOcc} set={setShowOcc} />
          <Toggle label="Wires" on={showWires} set={setShowWires} />
          <Toggle label="Velocity" on={showVel} set={setShowVel} />
          <Toggle label="Rig" on={showRig} set={setShowRig} />
        </div>
        <label className="slider">
          Corridor {radius.toFixed(2)} m
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={radius}
            onChange={(event) => setRadius(Number(event.target.value))}
          />
        </label>
        <div className="sun">
          <div className="modes" role="group" aria-label="Glare composite">
            {(["clean", "glare", "composite"] as GlareMode[]).map((mode) => (
              <button key={mode} className={glareMode === mode ? "on" : ""} onClick={() => setGlareMode(mode)} type="button">
                {mode}
              </button>
            ))}
          </div>
          <label>
            Sun az {azimuth.toFixed(0)}°
            <input
              type="range"
              min={0}
              max={360}
              value={azimuth}
              onChange={(event) => {
                setSunLocked(false);
                setAzimuth(Number(event.target.value));
              }}
            />
          </label>
          <label>
            El {elevation.toFixed(0)}°
            <input
              type="range"
              min={5}
              max={88}
              value={elevation}
              onChange={(event) => {
                setSunLocked(false);
                setElevation(Number(event.target.value));
              }}
            />
          </label>
          <button
            type="button"
            onClick={() => {
              if (cloud) {
                const angles = azElFromSun(cloud.sourceSun);
                setAzimuth(angles.azimuth);
                setElevation(angles.elevation);
              }
              setSunLocked(true);
            }}
            disabled={sunLocked}
          >
            Reset sun
          </button>
          <button type="button" onClick={() => void predict()} disabled={busy || !frame}>
            {busy ? "Predicting…" : "Predict glare"}
          </button>
        </div>
      </footer>
    </div>
  );
}

function EnuAxes() {
  return <axesHelper args={[2.2]} />;
}

function Toggle({ label, on, set }: { label: string; on: boolean; set: (value: boolean) => void }) {
  return (
    <button type="button" className={on ? "on" : ""} onClick={() => set(!on)} aria-pressed={on}>
      {label}
    </button>
  );
}

function HealthReadout({ health }: { health: Health | null }) {
  if (!health) return <div className="health">API …</div>;
  const vram = health.gpu.vram_used_bytes;
  const total = health.gpu.vram_total_bytes;
  const vramText =
    vram != null && total != null ? `${(vram / 1e9).toFixed(1)} / ${(total / 1e9).toFixed(1)} GB` : "no VRAM";
  const fps = health.throughput_fps;
  const totalMs = health.latency_ms.total_ms;
  return (
    <div className="health">
      <span>{health.gpu.name}</span>
      <span>{vramText}</span>
      <span>{totalMs != null ? `${totalMs.toFixed(1)} ms` : "no pass"}</span>
      <span>{fps != null ? `${fps.toFixed(2)} inf/s` : "—"}</span>
      <span>
        {health.model.loaded ? `${health.model.name} · ${health.model.init}` : `${health.model.name} cold`}
      </span>
    </div>
  );
}

function CameraStill({ imageUrl, glareUrl, mode }: { imageUrl: string; glareUrl: string; mode: GlareMode }) {
  const [clean, setClean] = useState<HTMLImageElement | null>(null);
  const [glare, setGlare] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    const image = new Image();
    image.onload = () => setClean(image);
    image.src = imageUrl;
  }, [imageUrl]);
  useEffect(() => {
    if (!glareUrl) {
      setGlare(null);
      return;
    }
    const image = new Image();
    image.onload = () => setGlare(image);
    image.src = glareUrl;
  }, [glareUrl]);
  const width = clean?.naturalWidth ?? 80;
  const height = clean?.naturalHeight ?? 60;
  return (
    <canvas
      width={width}
      height={height}
      ref={(node) => {
        if (!node || !clean) return;
        const ctx = node.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, width, height);
        if (mode === "glare") {
          ctx.fillStyle = "#050807";
          ctx.fillRect(0, 0, width, height);
          if (glare) ctx.drawImage(glare, 0, 0, width, height);
          return;
        }
        ctx.drawImage(clean, 0, 0, width, height);
        if (mode === "composite" && glare) {
          ctx.globalCompositeOperation = "lighter";
          ctx.drawImage(glare, 0, 0, width, height);
          ctx.globalCompositeOperation = "source-over";
        }
      }}
    />
  );
}

type OccupancyResponse = { occupancy_b64: string };
type InferResponse = {
  gaussian_b64: string;
  occupancy_b64: string;
  model: { init: string | null };
  glare: Record<string, { png_b64: string }>;
};

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

async function getBytes(path: string): Promise<ArrayBuffer> {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) throw new Error(await response.text());
  return response.arrayBuffer();
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

async function urlToB64(url: string): Promise<string> {
  const response = await fetch(url);
  const bytes = new Uint8Array(await response.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}
