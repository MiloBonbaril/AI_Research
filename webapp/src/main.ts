import modelsData from "../public/data/models.json";
import type { ModelsFile, ModelData } from "./types";
import { ArchitectureScene, type HoverInfo } from "./scene";
import { styleFor } from "./theme";

const data = modelsData as ModelsFile;

const canvas = document.getElementById("scene") as HTMLCanvasElement;
const tabsEl = document.getElementById("tabs")!;
const infoName = document.getElementById("info-name")!;
const infoParams = document.getElementById("info-params")!;
const infoDesc = document.getElementById("info-desc")!;
const infoCfg = document.getElementById("info-cfg")!;
const legendEl = document.getElementById("legend")!;
const tooltipEl = document.getElementById("tooltip")!;
const tooltipTitle = document.getElementById("tooltip-title")!;
const tooltipBody = document.getElementById("tooltip-body")!;

function hexColor(n: number): string {
  return "#" + n.toString(16).padStart(6, "0");
}

const scene = new ArchitectureScene(canvas, onHover);

function onHover(info: HoverInfo | null) {
  if (!info) {
    tooltipEl.classList.remove("visible");
    return;
  }
  tooltipTitle.textContent = info.title;
  tooltipBody.innerHTML = info.lines
    .slice(1)
    .map((l) => `<div class="t-line">${l}</div>`)
    .join("");
  tooltipEl.classList.add("visible");
}

function resize() {
  const rect = canvas.parentElement!.getBoundingClientRect();
  scene.resize(rect.width, rect.height);
}
window.addEventListener("resize", resize);

function renderInfo(model: ModelData) {
  infoName.textContent = model.name;
  infoParams.textContent = `${(model.total_params / 1e6).toFixed(1)}M parameters`;
  infoDesc.textContent = model.description;
  infoCfg.innerHTML = Object.entries(model.config)
    .map(([k, v]) => `<span>${k}: ${Array.isArray(v) ? `[${v.join(", ")}]` : v}</span>`)
    .join("");
}

function renderLegend(model: ModelData) {
  const kinds = new Set<string>(model.blocks.map((b) => b.kind));
  model.extras.forEach((e) => kinds.add(e.kind));
  legendEl.innerHTML = [...kinds]
    .map((k) => {
      const s = styleFor(k);
      return `<div class="row"><span class="swatch" style="background:${hexColor(s.color)};color:${hexColor(
        s.color
      )}"></span>${s.label}</div>`;
    })
    .join("");
}

function selectModel(id: string) {
  const model = data.models.find((m) => m.id === id);
  if (!model) return;
  [...tabsEl.children].forEach((el) =>
    el.classList.toggle("active", (el as HTMLElement).dataset.id === id)
  );
  renderInfo(model);
  renderLegend(model);
  scene.load(model);
}

data.models.forEach((model, i) => {
  const btn = document.createElement("button");
  btn.className = "tab";
  btn.textContent = model.name;
  btn.dataset.id = model.id;
  btn.addEventListener("click", () => selectModel(model.id));
  tabsEl.appendChild(btn);
  if (i === 0) requestAnimationFrame(() => selectModel(model.id));
});

resize();
