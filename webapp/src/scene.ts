import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { ModelData, BlockData, ExtraData } from "./types";
import { styleFor, THEME } from "./theme";

const TILE_SIZE = 1.7;
const TILE_GAP = 0.62;

function checkerTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const cell = size / 8;
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      ctx.fillStyle = (x + y) % 2 === 0 ? "#150c33" : "#0a0620";
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(14, 14);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeGeometry(shape: string): THREE.BufferGeometry {
  switch (shape) {
    case "octahedron":
      return new THREE.OctahedronGeometry(TILE_SIZE * 0.42, 0);
    case "torus":
      return new THREE.TorusGeometry(TILE_SIZE * 0.34, TILE_SIZE * 0.14, 12, 24);
    case "icosahedron":
      return new THREE.IcosahedronGeometry(TILE_SIZE * 0.45, 0);
    case "sphere":
      return new THREE.SphereGeometry(TILE_SIZE * 0.4, 24, 16);
    default:
      return new THREE.BoxGeometry(TILE_SIZE, 0.32, TILE_SIZE);
  }
}

function makeMesh(color: number, shape: string, heightScale: number): THREE.Mesh {
  const geo = makeGeometry(shape);
  if (shape === "box") geo.scale(1, heightScale, 1);
  const mat = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.45,
    metalness: 0.35,
    roughness: 0.3,
    transparent: true,
    opacity: 0.92,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = false;
  return mesh;
}

function edgeGlow(mesh: THREE.Mesh, color: number) {
  const edges = new THREE.EdgesGeometry(mesh.geometry);
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 })
  );
  mesh.add(line);
}

export interface HoverInfo {
  title: string;
  lines: string[];
}

export class ArchitectureScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private group = new THREE.Group();
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private pickables: THREE.Object3D[] = [];
  private hovered: THREE.Object3D | null = null;
  private onHover: (info: HoverInfo | null) => void;
  private clock = new THREE.Clock();
  private stars!: THREE.Points;

  constructor(canvas: HTMLCanvasElement, onHover: (info: HoverInfo | null) => void) {
    this.onHover = onHover;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
    this.camera.position.set(7, 6, 9);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 60;

    this.scene.background = new THREE.Color(THEME.bgBottom);
    this.scene.fog = new THREE.FogExp2(THEME.bgBottom, 0.028);

    this.scene.add(new THREE.AmbientLight(THEME.ambient, 1.1));
    const p1 = new THREE.PointLight(THEME.pointA, 40, 60);
    p1.position.set(8, 10, 6);
    const p2 = new THREE.PointLight(THEME.pointB, 30, 60);
    p2.position.set(-8, 4, -6);
    this.scene.add(p1, p2);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.MeshBasicMaterial({ map: checkerTexture(), transparent: true, opacity: 0.75 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -2;
    this.scene.add(ground);

    this.buildStarfield();
    this.scene.add(this.group);

    canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
    canvas.addEventListener("pointerleave", () => this.setHovered(null));

    this.animate();
  }

  private buildStarfield() {
    const count = 1400;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 40 + Math.random() * 140;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = Math.abs(r * Math.cos(phi)) * 0.6;
      positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color: 0xcbb6ff, size: 0.35, sizeAttenuation: true });
    this.stars = new THREE.Points(geo, mat);
    this.scene.add(this.stars);
  }

  resize(width: number, height: number) {
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  load(model: ModelData) {
    this.setHovered(null);
    this.clearGroup();
    this.pickables = [];

    const n = model.blocks.length;
    const avgParams = model.blocks.reduce((s, b) => s + b.params, 0) / Math.max(n, 1);

    model.blocks.forEach((block, i) => {
      const style = styleFor(block.kind);
      const heightScale = 0.6 + 0.9 * Math.min(2, block.params / Math.max(avgParams, 1));
      const mesh = makeMesh(style.color, style.shape, heightScale);
      mesh.position.set(0, i * TILE_GAP, 0);
      edgeGlow(mesh, style.color);
      mesh.userData = { info: this.blockInfo(block) };
      this.group.add(mesh);
      this.pickables.push(mesh);
    });

    // residual stream beam running through the stack
    const beamHeight = Math.max(n - 1, 0) * TILE_GAP + 1;
    const beam = new THREE.Mesh(
      new THREE.CylinderGeometry(0.04, 0.04, beamHeight, 8),
      new THREE.MeshBasicMaterial({ color: THEME.beam, transparent: true, opacity: 0.5 })
    );
    beam.position.set(0, (n - 1) * TILE_GAP * 0.5, 0);
    this.group.add(beam);

    // embedding cap (bottom) + lm head cap (top)
    const capGeo = new THREE.CylinderGeometry(TILE_SIZE * 0.62, TILE_SIZE * 0.62, 0.08, 24);
    const embd = new THREE.Mesh(
      capGeo,
      new THREE.MeshStandardMaterial({ color: 0xe0d4ff, emissive: 0x5b21b6, emissiveIntensity: 0.6 })
    );
    embd.position.set(0, -TILE_GAP, 0);
    this.group.add(embd);

    const head = new THREE.Mesh(
      capGeo,
      new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x0891b2, emissiveIntensity: 0.6 })
    );
    head.position.set(0, n * TILE_GAP, 0);
    this.group.add(head);

    // extras (Oneira's simulation head / world operators): branch sideways off
    // the top of the stack, like alternate timelines forking off the main line.
    const topY = n * TILE_GAP;
    model.extras.forEach((extra, i) => {
      const style = styleFor(extra.kind);
      const mesh = makeMesh(style.color, style.shape, 1);
      const angle = (i / Math.max(model.extras.length, 1)) * Math.PI * 0.9 - 0.45;
      const dist = 2.6 + i * 0.4;
      const ex = Math.sin(angle) * dist;
      const ey = topY + 1.1 + i * 0.5;
      const ez = Math.cos(angle) * dist - dist;
      mesh.position.set(ex, ey, ez);
      mesh.scale.setScalar(0.75);
      edgeGlow(mesh, style.color);
      mesh.userData = { info: this.extraInfo(extra) };
      this.group.add(mesh);
      this.pickables.push(mesh);

      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(0, topY, 0),
        new THREE.Vector3(ex * 0.4, topY + 0.6, ez * 0.4),
        new THREE.Vector3(ex, ey, ez)
      );
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(curve.getPoints(20)),
        new THREE.LineBasicMaterial({ color: style.color, transparent: true, opacity: 0.6 })
      );
      this.group.add(line);
    });

    // frame camera to fit the full stack + embedding/head caps + any extras branching off
    const extrasTop = model.extras.length
      ? topY + 1.1 + (model.extras.length - 1) * 0.5 + 1
      : topY;
    const contentTop = Math.max(topY, extrasTop) + 1;
    const contentBottom = -TILE_GAP - 0.6;
    const contentHeight = contentTop - contentBottom;
    const centerY = (contentTop + contentBottom) / 2;
    const dist = Math.max(9, contentHeight * 1.3);
    this.camera.position.set(dist * 0.6, centerY + contentHeight * 0.15, dist * 0.85);
    this.controls.target.set(0, centerY, 0);
    this.controls.update();
  }

  private blockInfo(block: BlockData): HoverInfo {
    const style = styleFor(block.kind);
    const lines = [
      `Layer ${block.index}`,
      style.label,
      `mixer: ${block.mixer_class}${block.mlp_class ? ` · mlp: ${block.mlp_class}` : ""}`,
      `params: ${(block.params / 1e6).toFixed(2)}M`,
    ];
    for (const [k, v] of Object.entries(block.detail)) {
      if (v !== null && v !== undefined) lines.push(`${k}: ${v}`);
    }
    return { title: style.label, lines };
  }

  private extraInfo(extra: ExtraData): HoverInfo {
    const style = styleFor(extra.kind);
    const lines = [
      extra.name,
      style.label,
      `class: ${extra.class}`,
      `params: ${(extra.params / 1e6).toFixed(2)}M`,
    ];
    for (const [k, v] of Object.entries(extra.detail)) {
      if (v !== null && v !== undefined) lines.push(`${k}: ${v}`);
    }
    return { title: `${extra.name} — ${style.label}`, lines };
  }

  private clearGroup() {
    for (const child of [...this.group.children]) {
      this.group.remove(child);
      if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
        child.geometry.dispose();
        const mat = child.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat.dispose();
      }
    }
  }

  private onPointerMove(e: PointerEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables, false);
    this.setHovered(hits.length ? hits[0].object : null);
  }

  private setHovered(obj: THREE.Object3D | null) {
    if (this.hovered === obj) return;
    if (this.hovered) (this.hovered as THREE.Mesh).scale.setScalar(this.hovered.userData.baseScale ?? 1);
    this.hovered = obj;
    if (obj) {
      obj.userData.baseScale = obj.userData.baseScale ?? obj.scale.x;
      (obj as THREE.Mesh).scale.setScalar(obj.userData.baseScale * 1.12);
      this.onHover(obj.userData.info as HoverInfo);
    } else {
      this.onHover(null);
    }
  }

  private animate = () => {
    requestAnimationFrame(this.animate);
    const dt = this.clock.getDelta();
    this.stars.rotation.y += dt * 0.006;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
