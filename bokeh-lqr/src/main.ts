/**
 * Discrete-time finite-horizon LQR phase-plot demo.
 * Relies on the global `Matrix` type + matrix helpers from linalg.ts
 * (compiled as a sibling global script, no module system involved)
 * and the global `Bokeh` object injected by the BokehJS CDN scripts.
 */

declare const Bokeh: any;

// ---------- Parsing ----------

/** Strips outer brackets/whitespace and returns the raw inner text. */
function stripBrackets(input: string): string {
  const trimmed = input.trim();
  const withoutBrackets =
    trimmed.startsWith("[") && trimmed.endsWith("]") ? trimmed.slice(1, -1) : trimmed;
  return withoutBrackets.trim();
}

/** Parses a MATLAB-style matrix literal, e.g. "[1, 0.1; 0, 1]". */
function parseMatrix(input: string): Matrix {
  const inner = stripBrackets(input);
  if (inner.length === 0) {
    throw new Error("Empty matrix");
  }
  const rows = inner.split(";").map((rowStr) => {
    const values = rowStr
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => {
        const n = Number(s);
        if (!Number.isFinite(n)) {
          throw new Error(`"${s}" is not a valid number`);
        }
        return n;
      });
    if (values.length === 0) {
      throw new Error("Empty row in matrix");
    }
    return values;
  });
  const width = rows[0].length;
  for (const row of rows) {
    if (row.length !== width) {
      throw new Error("Matrix rows must all have the same length");
    }
  }
  return rows;
}

/** Parses a flat list of numbers separated by "," and/or ";" into a column vector. */
function parseColumnVector(input: string): Matrix {
  const inner = stripBrackets(input);
  if (inner.length === 0) {
    throw new Error("Empty vector");
  }
  const values = inner
    .split(/[,;]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => {
      const n = Number(s);
      if (!Number.isFinite(n)) {
        throw new Error(`"${s}" is not a valid number`);
      }
      return n;
    });
  if (values.length === 0) {
    throw new Error("Empty vector");
  }
  return values.map((v) => [v]);
}

function parseScalar(input: string): number {
  const n = Number(input.trim());
  if (!Number.isFinite(n)) {
    throw new Error(`"${input}" is not a valid number`);
  }
  return n;
}

function assertSquare(M: Matrix, name: string): void {
  if (M.length === 0 || M.length !== M[0].length) {
    throw new Error(`${name} must be a square matrix`);
  }
}

function assertShape(M: Matrix, rows: number, cols: number, name: string): void {
  if (M.length !== rows || (M[0] ? M[0].length : 0) !== cols) {
    throw new Error(`${name} must be ${rows}x${cols}, got ${M.length}x${M[0] ? M[0].length : 0}`);
  }
}

// ---------- Riccati recursion ----------

interface RiccatiResult {
  P: Matrix[]; // P[0..N], cost-to-go matrices
  K: Matrix[]; // K[0..N-1], feedback gains (u_k = -K[k] * x_k)
}

/** Solves the discrete-time finite-horizon Riccati recursion backward from N to 0. */
function solveRiccati(A: Matrix, B: Matrix, Q: Matrix, Qf: Matrix, R: Matrix, N: number): RiccatiResult {
  const At = transpose(A);
  const Bt = transpose(B);

  const P: Matrix[] = new Array(N + 1);
  const K: Matrix[] = new Array(N);
  P[N] = Qf;

  for (let k = N - 1; k >= 0; k--) {
    const Pn = P[k + 1];
    const AtPn = matMul(At, Pn);
    const AtPnA = matMul(AtPn, A);
    const AtPnB = matMul(AtPn, B);
    const BtPnA = transpose(AtPnB);
    const S = matAdd(R, matMul(Bt, matMul(Pn, B))); // R + B^T P_{k+1} B
    const Sinv = matInverse(S);

    K[k] = matMul(Sinv, BtPnA); // (R + B^T P B)^-1 B^T P A
    const term = matMul(AtPnB, matMul(Sinv, BtPnA));
    P[k] = matAdd(Q, matSub(AtPnA, term));
  }

  return { P, K };
}

/** Forward-simulates x_{k+1} = (A - B K_k) x_k from x_0, k = 0..N-1. */
function simulateTrajectory(A: Matrix, B: Matrix, x0: Matrix, K: Matrix[], N: number): Matrix[] {
  const xs: Matrix[] = new Array(N + 1);
  xs[0] = x0;
  for (let k = 0; k < N; k++) {
    const u = scalarMul(-1, matMul(K[k], xs[k])); // u_k = -K_k x_k
    xs[k + 1] = matAdd(matMul(A, xs[k]), matMul(B, u));
  }
  return xs;
}

// ---------- UI wiring ----------

interface Inputs {
  A: HTMLInputElement;
  B: HTMLInputElement;
  Q: HTMLInputElement;
  Qf: HTMLInputElement;
  R: HTMLInputElement;
  x0: HTMLInputElement;
  NMax: HTMLInputElement;
  N: HTMLInputElement;
  penalize: HTMLInputElement;
  alpha: HTMLInputElement;
}

function getEl<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
}

const SURFACE = "#fcfcfb";
const GRIDLINE = "#e1e0d9";
const INK = "#0b0b0b";
const SERIES_1 = "#2a78d6"; // categorical slot 1 — the trajectory / cost curve
const SERIES_2 = "#eb6834"; // categorical slot 2 — the selected horizon

/** Shared figure chrome so both plots read as one system. */
function makeFigure(opts: Record<string, unknown>): any {
  const fig = Bokeh.Plotting.figure(
    Object.assign(
      {
        toolbar_location: "above",
        // Fills whatever box the CSS gives it, so the 16:9 frame drives sizing.
        sizing_mode: "stretch_both",
        background_fill_color: SURFACE,
        border_fill_color: SURFACE,
      },
      opts
    )
  );
  fig.xgrid.grid_line_color = GRIDLINE;
  fig.ygrid.grid_line_color = GRIDLINE;
  fig.title.text_color = INK;
  return fig;
}

let phaseView: any = null;
let costView: any = null;

/**
 * Tear a plot down completely. Removing the view detaches its DOM, but the
 * Document it was shown into stays registered in `Bokeh.documents` forever —
 * so without this the registry (and everything it retains) grows by one per
 * plot on every re-render.
 */
function destroyView(view: any): null {
  if (!view) return null;
  const doc = view.model != null ? view.model.document : null;
  view.remove();
  if (doc != null) {
    doc.clear();
    const registry = Bokeh.documents;
    const i = registry.indexOf(doc);
    if (i >= 0) registry.splice(i, 1);
  }
  return null;
}

async function renderPhasePlot(
  xs: number[],
  ys: number[],
  steps: number[],
  title: string
): Promise<void> {
  const container = getEl<HTMLDivElement>("phase-plot");
  phaseView = destroyView(phaseView);
  container.innerHTML = "";

  const source = new Bokeh.ColumnDataSource({ data: { x: xs, y: ys, k: steps } });

  const hover = new Bokeh.HoverTool({
    tooltips: [
      ["step k", "@k"],
      ["position", "@x{0.000}"],
      ["velocity", "@y{0.000}"],
    ],
  });

  const fig = makeFigure({
    title: title,
    tools: [hover, "pan", "wheel_zoom", "box_zoom", "reset", "save"],
    x_axis_label: "Position",
    y_axis_label: "Velocity",
    x_range: new Bokeh.Range1d({ start: -5, end: 5 }),
    y_range: new Bokeh.Range1d({ start: -5, end: 5 }),
  });

  fig.line({ field: "x" }, { field: "y" }, {
    source: source,
    line_width: 2,
    line_color: SERIES_1,
    line_join: "round",
    line_cap: "round",
  });
  fig.scatter({ field: "x" }, { field: "y" }, {
    source: source,
    marker: "circle",
    size: 8,
    fill_color: SERIES_1,
    line_color: SURFACE,
    line_width: 2,
  });

  phaseView = await Bokeh.Plotting.show(fig, container);
}

async function renderCostPlot(
  horizons: number[],
  costs: number[],
  selectedN: number,
  selectedCost: number,
  penalized: boolean
): Promise<void> {
  const container = getEl<HTMLDivElement>("cost-plot");
  costView = destroyView(costView);
  container.innerHTML = "";

  const curve = new Bokeh.ColumnDataSource({ data: { n: horizons, j: costs } });
  const marker = new Bokeh.ColumnDataSource({
    data: { n: [selectedN], j: [selectedCost], label: [`N = ${selectedN}`] },
  });

  const hover = new Bokeh.HoverTool({
    tooltips: [
      ["horizon N", "@n"],
      [penalized ? "total cost" : "cost-to-go", "@j{0.0000}"],
    ],
    mode: "vline",
  });

  const fig = makeFigure({
    title: penalized
      ? "Total cost vs horizon — x₀ᵀPₖx₀ + αN, so each extra step is charged for"
      : "Cost-to-go vs horizon — Pₖ from one sweep solves the horizon-(Nₘₐₓ−k) problem",
    tools: [hover, "pan", "wheel_zoom", "box_zoom", "reset", "save"],
    x_axis_label: "Horizon N",
    y_axis_label: penalized ? "Total cost x₀ᵀPx₀ + αN" : "Cost-to-go x₀ᵀPx₀",
  });

  fig.line({ field: "n" }, { field: "j" }, {
    source: curve,
    line_width: 2,
    line_color: SERIES_1,
    line_join: "round",
    line_cap: "round",
  });

  // The horizon currently selected on the slider — ties this plot to the phase plot above.
  fig.scatter({ field: "n" }, { field: "j" }, {
    source: marker,
    marker: "circle",
    size: 11,
    fill_color: SERIES_2,
    line_color: SURFACE,
    line_width: 2,
  });
  // Flip the label inward near the right edge so it never runs off the plot.
  const nearRightEdge = horizons.length > 1 && selectedN > horizons[horizons.length - 1] * 0.85;
  fig.add_layout(
    new Bokeh.LabelSet({
      x: { field: "n" },
      y: { field: "j" },
      text: { field: "label" },
      source: marker,
      x_offset: nearRightEdge ? -8 : 8,
      y_offset: 6,
      text_align: nearRightEdge ? "right" : "left",
      text_font_size: "11px",
      text_color: "#52514e",
    })
  );

  costView = await Bokeh.Plotting.show(fig, container);
}

function setStatus(message: string, isError: boolean): void {
  const status = getEl<HTMLDivElement>("status");
  status.textContent = message;
  status.className = isError ? "status status--error" : "status status--ok";
}

/** Upper bound on N_max — keeps one sweep (and the curve it draws) cheap. */
const NMAX_CAP = 5000;

function update(inputs: Inputs): void {
  let A: Matrix, B: Matrix, Q: Matrix, Qf: Matrix, x0: Matrix, R: Matrix;
  let N: number, NMax: number, alpha: number;
  const NLabel = getEl<HTMLSpanElement>("N-value");
  NLabel.textContent = inputs.N.value;

  // The α field only exists while the penalty is switched on.
  const penalized = inputs.penalize.checked;
  getEl<HTMLDivElement>("alpha-field").hidden = !penalized;

  try {
    A = parseMatrix(inputs.A.value);
    B = parseColumnVector(inputs.B.value);
    Q = parseMatrix(inputs.Q.value);
    Qf = parseMatrix(inputs.Qf.value);
    x0 = parseColumnVector(inputs.x0.value);
    R = [[parseScalar(inputs.R.value)]];

    alpha = 0;
    if (penalized) {
      alpha = parseScalar(inputs.alpha.value);
      if (!(alpha > 0)) {
        throw new Error("α must be a positive number");
      }
    }

    NMax = parseScalar(inputs.NMax.value);
    if (!Number.isInteger(NMax) || NMax < 1) {
      throw new Error("N_max must be a positive integer");
    }
    if (NMax > NMAX_CAP) {
      throw new Error(`N_max is capped at ${NMAX_CAP}`);
    }
    // The slider spans 1..N_max, so widening or narrowing N_max reshapes it.
    inputs.N.max = String(NMax);
    if (Number(inputs.N.value) > NMax) {
      inputs.N.value = String(NMax);
    }
    N = Number(inputs.N.value);
    NLabel.textContent = String(N);

    const n = x0.length;
    if (n !== 2) {
      throw new Error("x₀ must have 2 entries — the phase plot needs a position and a velocity");
    }
    assertSquare(A, "A");
    assertShape(A, n, n, "A");
    assertShape(B, n, 1, "B");
    assertSquare(Q, "Q");
    assertShape(Q, n, n, "Q");
    assertSquare(Qf, "Qf");
    assertShape(Qf, n, n, "Qf");
    if (!Number.isInteger(N) || N < 1) {
      throw new Error("Horizon N must be a positive integer");
    }
  } catch (err) {
    setStatus(`Input error: ${(err as Error).message}`, true);
    return;
  }

  try {
    // One backward pass over the widest horizon requested. Because the
    // recursion starts from P_{Nmax} = Qf, the matrix P_k is exactly the
    // cost-to-go of a problem with Nmax - k steps left — so this single sweep
    // already contains the solution to *every* horizon from 0 to Nmax, and both
    // plots below are just different slices of it.
    const { P, K } = solveRiccati(A, B, Q, Qf, R, NMax);

    const x0T = transpose(x0);
    const costAt = (k: number) => matMul(matMul(x0T, P[k]), x0)[0][0];

    // Cost-to-go curve: horizon h maps to P[NMax - h]. Starts at h = 1, since
    // h = 0 is the degenerate no-move case where the cost is just x₀ᵀQ_fx₀.
    // With the penalty on, each horizon also pays α per step it uses.
    const horizons: number[] = [];
    const costs: number[] = [];
    for (let h = 1; h <= NMax; h++) {
      horizons.push(h);
      costs.push(costAt(NMax - h) + alpha * h);
    }

    // Phase trajectory for the selected horizon N. Its gain at step j is the
    // gain of the full sweep at the index with the same number of steps
    // remaining: NMax - N + j.
    const gains = K.slice(NMax - N);
    const xs = simulateTrajectory(A, B, x0, gains, N);

    const positions = xs.map((x) => x[0][0]);
    const velocities = xs.map((x) => x[1][0]);
    const steps = xs.map((_, k) => k);

    const baseCost = costAt(NMax - N);
    const penalty = alpha * N;
    const cost = baseCost + penalty;
    const title = penalized
      ? `Phase plot — J(x₀) = x₀ᵀP₀x₀ + αN = ${baseCost.toFixed(4)} + ${penalty.toFixed(4)} = ${cost.toFixed(4)}`
      : `Phase plot — cost-to-go J(x₀) = x₀ᵀP₀x₀ = ${cost.toFixed(4)}`;

    renderPhasePlot(positions, velocities, steps, title);
    renderCostPlot(horizons, costs, N, cost, penalized);
    setStatus("OK", false);
  } catch (err) {
    setStatus(`Computation error: ${(err as Error).message}`, true);
  }
}

function main(): void {
  const inputs: Inputs = {
    A: getEl("input-A"),
    B: getEl("input-B"),
    Q: getEl("input-Q"),
    Qf: getEl("input-Qf"),
    R: getEl("input-R"),
    x0: getEl("input-x0"),
    NMax: getEl("input-Nmax"),
    N: getEl("input-N"),
    penalize: getEl("input-penalize"),
    alpha: getEl("input-alpha"),
  };

  let debounceHandle: number | undefined;
  const scheduleUpdate = () => {
    if (debounceHandle !== undefined) window.clearTimeout(debounceHandle);
    debounceHandle = window.setTimeout(() => update(inputs), 150);
  };

  (Object.keys(inputs) as (keyof Inputs)[]).forEach((key) => {
    inputs[key].addEventListener("input", scheduleUpdate);
  });
  // Checkboxes toggle via keyboard/label as well, which some browsers report
  // only as "change".
  inputs.penalize.addEventListener("change", scheduleUpdate);

  update(inputs);
}

document.addEventListener("DOMContentLoaded", main);
