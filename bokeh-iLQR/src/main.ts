/**
 * UI wiring: read the problem setup, run iLQR, and scrub through iterations.
 */

interface AppState {
  result: IlqrResult | null;
  win: WorldWindow | null;
  accelDomain: [number, number] | null;
  steerDomain: [number, number] | null;
  iter: number;
}

const state: AppState = {
  result: null,
  win: null,
  accelDomain: null,
  steerDomain: null,
  iter: 0,
};

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error("Missing element #" + id);
  return node as T;
}

const x0Input = el<HTMLInputElement>("x0-input");
const radiusInput = el<HTMLInputElement>("radius-input");
const speedInput = el<HTMLInputElement>("speed-input");
const horizonInput = el<HTMLInputElement>("horizon-input");
const maxIterInput = el<HTMLInputElement>("maxiter-input");
const reguInput = el<HTMLInputElement>("regu-input");
const runButton = el<HTMLButtonElement>("run-button");
const errorBox = el<HTMLDivElement>("error-box");
const statusBox = el<HTMLSpanElement>("status-box");
const slider = el<HTMLInputElement>("iter-slider");
const sliderReadout = el<HTMLSpanElement>("iter-readout");
const sliderMaxLabel = el<HTMLSpanElement>("slider-max");
const trajHost = el<HTMLDivElement>("traj-plot");
const costHost = el<HTMLDivElement>("cost-plot");
const accelHost = el<HTMLDivElement>("accel-plot");
const steerHost = el<HTMLDivElement>("steer-plot");
const dataButton = el<HTMLButtonElement>("data-button");
const dataPanel = el<HTMLDivElement>("data-panel");
const panelClose = el<HTMLButtonElement>("panel-close");
const mathButton = el<HTMLButtonElement>("math-button");
const mathPanel = el<HTMLDivElement>("math-panel");
const mathClose = el<HTMLButtonElement>("math-close");
const mathParams = el<HTMLDivElement>("math-params");
const panelSub = el<HTMLSpanElement>("panel-sub");
const traceTableBody = el<HTMLTableSectionElement>("trace-table-body");
const trajTableBody = el<HTMLTableSectionElement>("traj-table-body");
const trajTableTitle = el<HTMLHeadingElement>("traj-table-title");

/** Parse "[1, 0, 0, 1, 0]" (brackets and separators are all optional). */
function parseState(text: string): Vec {
  const cleaned = text.replace(/[\[\]()]/g, " ").trim();
  const parts = cleaned.split(/[\s,;]+/).filter(function (p) {
    return p.length > 0;
  });
  if (parts.length !== N_X) {
    throw new Error(
      "Initial state needs " + N_X + " numbers [x, y, heading, speed, steering]" +
        ", got " + parts.length
    );
  }
  return parts.map(function (p) {
    const v = Number(p);
    if (!isFinite(v)) throw new Error('"' + p + '" is not a number');
    return v;
  });
}

function parseInt_(input: HTMLInputElement, name: string, min: number): number {
  const v = Number(input.value);
  if (!isFinite(v) || Math.floor(v) !== v || v < min) {
    throw new Error(name + " must be an integer >= " + min);
  }
  return v;
}

function parseNumber(
  input: HTMLInputElement, name: string, min: number, allowMin: boolean
): number {
  const v = Number(input.value);
  const ok = isFinite(v) && (allowMin ? v >= min : v > min);
  if (input.value.trim() === "" || !ok) {
    throw new Error(name + " must be a number " + (allowMin ? ">= " : "> ") + min);
  }
  return v;
}

function run(): void {
  errorBox.textContent = "";
  errorBox.classList.remove("visible");

  let x0: Vec;
  let N: number;
  let maxIter: number;
  let reguInit: number;
  let radius: number;
  let speed: number;
  try {
    x0 = parseState(x0Input.value);
    radius = parseNumber(radiusInput, "Target radius", 0, true);
    speed = parseNumber(speedInput, "Target speed", -Infinity, true);
    N = parseInt_(horizonInput, "Horizon N", 2);
    maxIter = parseInt_(maxIterInput, "Max iterations", 1);
    reguInit = parseNumber(reguInput, "Initial regularization", 0, false);
  } catch (e) {
    errorBox.textContent = (e as Error).message;
    errorBox.classList.add("visible");
    return;
  }

  setCostTargets(radius, speed);

  let result: IlqrResult;
  const t0 = performance.now();
  try {
    result = runIlqr(x0, N, maxIter, reguInit, 12345);
  } catch (e) {
    errorBox.textContent = "iLQR failed: " + (e as Error).message;
    errorBox.classList.add("visible");
    return;
  }
  const elapsed = performance.now() - t0;

  state.result = result;
  state.win = trajectoryWindow(result.xTrace);
  state.accelDomain = controlDomain(result.uTrace, 0);
  state.steerDomain = controlDomain(result.uTrace, 1);
  state.iter = 0;

  const lastIndex = result.costTrace.length - 1;
  slider.min = "0";
  slider.max = String(lastIndex);
  slider.value = "0";
  slider.disabled = lastIndex === 0;
  sliderMaxLabel.textContent = String(lastIndex);

  statusBox.textContent =
    (result.converged
      ? "converged after " + result.iterationsRun + " iterations"
      : "ran " + result.iterationsRun + " iterations, no early termination") +
    " · cost " + formatCost(result.costTrace[0]) +
    " → " + formatCost(result.costTrace[lastIndex]) +
    " · " + elapsed.toFixed(0) + " ms";

  fillTraceTable(result);
  render();
}

function addRow(body: HTMLElement, cells: string[]): void {
  const tr = document.createElement("tr");
  for (let c = 0; c < cells.length; c++) {
    const td = document.createElement("td");
    td.textContent = cells[c];
    tr.appendChild(td);
  }
  body.appendChild(tr);
}

function fillTraceTable(result: IlqrResult): void {
  clearNode(traceTableBody);
  for (let i = 0; i < result.costTrace.length; i++) {
    addRow(traceTableBody, [
      String(i),
      formatCost(result.costTrace[i]),
      formatCost(result.reguTrace[i]),
      i === 0 ? "—" : result.reduRatioTrace[i].toFixed(3),
    ]);
  }
}

function fillTrajTable(result: IlqrResult, iter: number): void {
  clearNode(trajTableBody);
  const xTrj = result.xTrace[iter];
  const uTrj = result.uTrace[iter];
  trajTableTitle.textContent = "Trajectory at iteration " + iter;
  for (let n = 0; n < xTrj.length; n++) {
    const hasU = n < uTrj.length;
    addRow(trajTableBody, [
      String(n),
      xTrj[n][0].toFixed(3),
      xTrj[n][1].toFixed(3),
      xTrj[n][2].toFixed(3),
      xTrj[n][3].toFixed(3),
      xTrj[n][4].toFixed(3),
      hasU ? uTrj[n][0].toFixed(3) : "—",
      hasU ? uTrj[n][1].toFixed(3) : "—",
    ]);
  }
}

function seek(i: number): void {
  state.iter = i;
  slider.value = String(i);
  render();
}

function render(): void {
  const result = state.result;
  if (!result || !state.win) return;

  const iter = Math.min(state.iter, result.xTrace.length - 1);
  const lastIndex = result.costTrace.length - 1;
  sliderReadout.textContent = String(iter);

  renderTrajectory(
    trajHost,
    result.xTrace[iter],
    state.win,
    trajHost.clientWidth,
    trajHost.clientHeight
  );

  renderLineChart(costHost, {
    values: result.costTrace,
    xDomain: [0, lastIndex],
    pointerIndex: iter,
    xAxisLabel: "iteration",
    format: formatCost,
    ariaLabel: "Total cost per iLQR iteration",
    tooltip: function (i: number) {
      return {
        title: "Iteration " + i,
        rows: [
          ["Total cost", formatCost(result.costTrace[i])],
          ["Regularization", formatCost(result.reguTrace[i])],
          ["Reduction ratio", i === 0 ? "—" : result.reduRatioTrace[i].toFixed(3)],
        ],
      };
    },
    onSeek: seek,
    width: costHost.clientWidth,
    height: costHost.clientHeight,
  });

  const uTrj = result.uTrace[iter];
  const controls: {
    host: HTMLElement;
    comp: number;
    domain: [number, number];
    name: string;
    unit: string;
  }[] = [
    { host: accelHost, comp: 0, domain: state.accelDomain!, name: "Acceleration", unit: "m/s²" },
    { host: steerHost, comp: 1, domain: state.steerDomain!, name: "Steering velocity", unit: "rad/s" },
  ];

  for (let c = 0; c < controls.length; c++) {
    const spec = controls[c];
    const values: number[] = [];
    for (let n = 0; n < uTrj.length; n++) values.push(uTrj[n][spec.comp]);
    renderLineChart(spec.host, {
      values,
      xDomain: [0, Math.max(1, values.length - 1)],
      yDomain: spec.domain,
      xAxisLabel: "timestep n",
      zeroLine: true,
      format: formatControl,
      ariaLabel: spec.name + " control trajectory at iteration " + iter,
      tooltip: function (i: number) {
        return {
          title: "Timestep " + i,
          rows: [
            [spec.name, formatControl(values[i]) + " " + spec.unit],
            ["Time", (i * DT).toFixed(1) + " s"],
          ],
        };
      },
      width: spec.host.clientWidth,
      height: spec.host.clientHeight,
    });
  }

  if (dataPanel.classList.contains("visible")) {
    fillTrajTable(result, iter);
    panelSub.textContent = "iteration " + iter + " of " + lastIndex;
  }
}

slider.addEventListener("input", function () {
  state.iter = Number(slider.value);
  render();
});

runButton.addEventListener("click", run);

[x0Input, radiusInput, speedInput, horizonInput, maxIterInput, reguInput].forEach(
  function (input) {
    input.addEventListener("keydown", function (ev: KeyboardEvent) {
      if (ev.key === "Enter") run();
    });
  }
);

/* Data panel ---------------------------------------------------------- */
function openPanel(): void {
  const result = state.result;
  if (!result) return;
  dataPanel.classList.add("visible");
  fillTrajTable(result, state.iter);
  panelSub.textContent =
    "iteration " + state.iter + " of " + (result.costTrace.length - 1);
}

function closePanel(): void {
  dataPanel.classList.remove("visible");
}

dataButton.addEventListener("click", openPanel);
panelClose.addEventListener("click", closePanel);
dataPanel.addEventListener("pointerdown", function (ev: Event) {
  if (ev.target === dataPanel) closePanel();
});

/* Math panel ---------------------------------------------------------- */

/** The symbols in the formulas, filled in with the values of the current run. */
function fillMathParams(): void {
  const rows: string[][] = [
    ["r", String(costParams.r)],
    ["v★", String(costParams.vTarget)],
    ["Δt", String(DT) + " s"],
    ["w", String(CONTROL_WEIGHT)],
    ["ε", "1e−6"],
    ["N", horizonInput.value],
  ];
  clearNode(mathParams);
  for (let i = 0; i < rows.length; i++) {
    const cell = document.createElement("div");
    const name = document.createElement("span");
    name.textContent = rows[i][0];
    const value = document.createElement("b");
    value.textContent = rows[i][1];
    cell.appendChild(name);
    cell.appendChild(value);
    mathParams.appendChild(cell);
  }
}

/**
 * Reveal the formulas once MathJax has typeset them. The timeout is a fallback:
 * if MathJax is missing the LaTeX source shows rather than nothing at all.
 */
function revealMath(): void {
  const scroll = document.querySelector(".math-scroll");
  if (scroll) scroll.classList.add("ready");
}
const mathJax = (window as any).MathJax;
if (mathJax && mathJax.startup && mathJax.startup.promise) {
  mathJax.startup.promise.then(revealMath);
}
window.setTimeout(revealMath, 3000);

mathButton.addEventListener("click", function () {
  fillMathParams();
  mathPanel.classList.add("visible");
});
mathClose.addEventListener("click", function () {
  mathPanel.classList.remove("visible");
});
mathPanel.addEventListener("pointerdown", function (ev: Event) {
  if (ev.target === mathPanel) mathPanel.classList.remove("visible");
});

document.addEventListener("keydown", function (ev: KeyboardEvent) {
  if (ev.key === "Escape") {
    closePanel();
    mathPanel.classList.remove("visible");
  }
});

/* Keep the SVGs matched to their boxes ------------------------------- */
let resizeFrame = 0;
window.addEventListener("resize", function () {
  if (resizeFrame) cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(function () {
    resizeFrame = 0;
    render();
  });
});

// Run once with the notebook's own setup so the page is not empty on load.
run();
