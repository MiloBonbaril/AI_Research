export interface KindStyle {
  color: number;
  shape: "box" | "octahedron" | "torus" | "icosahedron" | "sphere";
  label: string;
}

// Palette tuned to the "5D Chess with Multiverse Time Travel" look: deep-space
// violets/blues as the base, each mixer kind gets its own neon accent + shape
// so the silhouette alone tells you what changed between architectures.
export const KIND_STYLES: Record<string, KindStyle> = {
  "full-attention": { color: 0x22d3ee, shape: "box", label: "Full Causal Attention" },
  "ternary-attention": { color: 0x34d399, shape: "icosahedron", label: "Ternary Attention (BitLinear)" },
  "global-attention": { color: 0x38bdf8, shape: "box", label: "Global Attention" },
  "local-attention": { color: 0x3b82f6, shape: "box", label: "Local (Sliding-Window) Attention" },
  "linear-attention": { color: 0xfb7185, shape: "octahedron", label: "Linear Attention" },
  gla: { color: 0xa855f7, shape: "torus", label: "Gated Linear Attention (recurrent state)" },
  csa: { color: 0xf59e0b, shape: "box", label: "Compressed Sparse Attention (top-k)" },
  hca: { color: 0xfbbf24, shape: "box", label: "Heavily Compressed Attention (dense)" },
  "simulation-head": { color: 0xfde68a, shape: "sphere", label: "World-Simulation Head" },
  "world-operator": { color: 0xe879f9, shape: "sphere", label: "World Operator F (future horizon)" },
  projection: { color: 0x94a3b8, shape: "box", label: "Projection" },
  unknown: { color: 0x64748b, shape: "box", label: "Unknown" },
};

export function styleFor(kind: string): KindStyle {
  return KIND_STYLES[kind] ?? KIND_STYLES.unknown;
}

export const THEME = {
  bgTop: 0x0a0620,
  bgBottom: 0x030110,
  gridLineA: 0x7c3aed,
  gridLineB: 0x1e1b4b,
  beam: 0x67e8f9,
  ambient: 0x2a1a5e,
  pointA: 0x8b5cf6,
  pointB: 0x22d3ee,
};
