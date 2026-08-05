/**
 * SVG rendering for the four plots. No chart library: everything is drawn from
 * the trajectory, control and cost traces directly.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(name: string, attrs: { [k: string]: string | number }): SVGElement {
  const el = document.createElementNS(SVG_NS, name) as SVGElement;
  for (const key in attrs) {
    if (Object.prototype.hasOwnProperty.call(attrs, key)) {
      el.setAttribute(key, String(attrs[key]));
    }
  }
  return el;
}

function clearNode(node: Element): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** Tick positions on a 1-2-5 ladder covering [min, max]. */
function niceTicks(min: number, max: number, target: number): number[] {
  if (!isFinite(min) || !isFinite(max)) return [0];
  if (max - min < 1e-12) return [min];
  const rawStep = (max - min) / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  let step: number;
  if (norm <= 1) step = 1;
  else if (norm <= 2) step = 2;
  else if (norm <= 5) step = 5;
  else step = 10;
  step *= mag;

  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step * 1e-6; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return ticks;
}

/** Integer tick positions for the iteration / timestep axes. */
function integerTicks(min: number, max: number, target: number): number[] {
  const span = Math.max(1, max - min);
  let step = Math.max(1, Math.ceil(span / target));
  const mag = Math.pow(10, Math.floor(Math.log10(step)));
  const norm = step / mag;
  if (norm <= 1) step = mag;
  else if (norm <= 2) step = 2 * mag;
  else if (norm <= 5) step = 5 * mag;
  else step = 10 * mag;

  const ticks: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max; v += step) ticks.push(v);
  if (ticks.length === 0 || ticks[ticks.length - 1] !== max) ticks.push(max);
  return ticks;
}

function formatCost(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

/** Signed, fixed-width-ish formatting for the control values. */
function formatControl(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

/* --------------------------------------------------------------------------
 * Trajectory plot
 * ----------------------------------------------------------------------- */

interface WorldWindow {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.round(q * (sorted.length - 1)))
  );
  return sorted[idx];
}

/**
 * One world window shared by every iteration, so the view does not jump around
 * while the slider moves.
 *
 * It always contains the target circle, the initial state and the final
 * trajectory. Intermediate iterations are included only up to a high quantile:
 * a single early iteration that flies off would otherwise shrink every other
 * one to a squiggle.
 */
function trajectoryWindow(xTrace: Mat[]): WorldWindow {
  const r = costParams.r;
  let xMin = -r;
  let xMax = r;
  let yMin = -r;
  let yMax = r;

  // Must-show: the initial state and the converged trajectory
  const mustShow: Mat = [xTrace[0][0]].concat(xTrace[xTrace.length - 1]);
  for (let n = 0; n < mustShow.length; n++) {
    xMin = Math.min(xMin, mustShow[n][0]);
    xMax = Math.max(xMax, mustShow[n][0]);
    yMin = Math.min(yMin, mustShow[n][1]);
    yMax = Math.max(yMax, mustShow[n][1]);
  }

  // Nice-to-show: the bulk of every intermediate iteration
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < xTrace.length; i++) {
    const trj = xTrace[i];
    for (let n = 0; n < trj.length; n++) {
      xs.push(trj[n][0]);
      ys.push(trj[n][1]);
    }
  }
  xs.sort(function (a, b) { return a - b; });
  ys.sort(function (a, b) { return a - b; });
  xMin = Math.min(xMin, quantile(xs, 0.02));
  xMax = Math.max(xMax, quantile(xs, 0.98));
  yMin = Math.min(yMin, quantile(ys, 0.02));
  yMax = Math.max(yMax, quantile(ys, 0.98));

  // Room for the car body (2.0 x 1.0) plus a margin
  const pad = 1.4;
  return {
    xMin: xMin - pad,
    xMax: xMax + pad,
    yMin: yMin - pad,
    yMax: yMax + pad,
  };
}

const CAR_W = 2.0;
const CAR_H = 1.0;

function renderTrajectory(
  host: HTMLElement,
  xTrj: Mat,
  win: WorldWindow,
  width: number,
  height: number
): void {
  clearNode(host);
  const margin = { top: 12, right: 14, bottom: 26, left: 40 };
  const plotW = Math.max(10, width - margin.left - margin.right);
  const plotH = Math.max(10, height - margin.top - margin.bottom);

  // Equal aspect: one scale for both axes, centred in the plot area
  const scale = Math.min(
    plotW / (win.xMax - win.xMin),
    plotH / (win.yMax - win.yMin)
  );
  const cx = (win.xMin + win.xMax) / 2;
  const cy = (win.yMin + win.yMax) / 2;
  const originX = margin.left + plotW / 2;
  const originY = margin.top + plotH / 2;
  const sx = (x: number) => originX + (x - cx) * scale;
  const sy = (y: number) => originY - (y - cy) * scale;

  const svg = svgEl("svg", {
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Planned vehicle trajectory in the x-y plane",
  });

  // Visible world extent after the equal-aspect fit
  const halfW = plotW / 2 / scale;
  const halfH = plotH / 2 / scale;

  const grid = svgEl("g", { class: "grid" });
  const xTicks = niceTicks(cx - halfW, cx + halfW, 7);
  const yTicks = niceTicks(cy - halfH, cy + halfH, 5);
  for (let i = 0; i < xTicks.length; i++) {
    const px = sx(xTicks[i]);
    grid.appendChild(
      svgEl("line", { x1: px, y1: margin.top, x2: px, y2: margin.top + plotH })
    );
    const label = svgEl("text", {
      x: px,
      y: margin.top + plotH + 17,
      class: "tick",
      "text-anchor": "middle",
    });
    label.textContent = String(xTicks[i]);
    svg.appendChild(label);
  }
  for (let i = 0; i < yTicks.length; i++) {
    const py = sy(yTicks[i]);
    grid.appendChild(
      svgEl("line", { x1: margin.left, y1: py, x2: margin.left + plotW, y2: py })
    );
    const label = svgEl("text", {
      x: margin.left - 7,
      y: py + 4,
      class: "tick",
      "text-anchor": "end",
    });
    label.textContent = String(yTicks[i]);
    svg.appendChild(label);
  }
  svg.insertBefore(grid, svg.firstChild);

  // An iteration that overshoots is clipped to the plot area rather than drawn
  // across the axis labels.
  const clipId = "traj-clip";
  const defs = svgEl("defs", {});
  const clip = svgEl("clipPath", { id: clipId });
  clip.appendChild(
    svgEl("rect", { x: margin.left, y: margin.top, width: plotW, height: plotH })
  );
  defs.appendChild(clip);
  svg.appendChild(defs);
  const plotLayer = svgEl("g", { "clip-path": "url(#" + clipId + ")" });

  // Target circle (the cost pulls the car onto this)
  plotLayer.appendChild(
    svgEl("circle", {
      cx: sx(0),
      cy: sy(0),
      r: costParams.r * scale,
      class: "target-circle",
    })
  );

  // Car bodies, thinned out so they stay readable at long horizons and do not
  // pile into a solid block where the car is moving slowly.
  const bodies = svgEl("g", { class: "car-bodies" });
  const step = Math.max(1, Math.round(xTrj.length / 26));
  let lastX = NaN;
  let lastY = NaN;
  for (let n = 0; n < xTrj.length; n += step) {
    const px = sx(xTrj[n][0]);
    const py = sy(xTrj[n][1]);
    if (n > 0 && Math.hypot(px - lastX, py - lastY) < 14) continue;
    bodies.appendChild(carPolygon(xTrj[n], sx, sy));
    lastX = px;
    lastY = py;
  }
  bodies.appendChild(carPolygon(xTrj[xTrj.length - 1], sx, sy));
  plotLayer.appendChild(bodies);

  // Planned path, drawn ".-" style: a line through every knot plus a dot at
  // each discrete state x[n].
  let d = "";
  for (let n = 0; n < xTrj.length; n++) {
    d += (n === 0 ? "M" : "L") + sx(xTrj[n][0]) + " " + sy(xTrj[n][1]);
  }
  plotLayer.appendChild(svgEl("path", { d, class: "traj-line" }));

  const knots = svgEl("g", { class: "traj-knots" });
  for (let n = 0; n < xTrj.length; n++) {
    knots.appendChild(
      svgEl("circle", { cx: sx(xTrj[n][0]), cy: sy(xTrj[n][1]), r: 2.4 })
    );
  }
  plotLayer.appendChild(knots);

  // Endpoint markers, direct-labelled. Early iterations barely move, so the two
  // labels are only drawn separately once the endpoints are far enough apart.
  const last = xTrj[xTrj.length - 1];
  const x0px = sx(xTrj[0][0]);
  const y0px = sy(xTrj[0][1]);
  const x1px = sx(last[0]);
  const y1px = sy(last[1]);
  const separated = Math.hypot(x1px - x0px, y1px - y0px) > 34;

  // Push each label radially outward from the middle of the path, so it lands
  // off to the side of the curve rather than on top of it.
  let midX = 0;
  let midY = 0;
  for (let n = 0; n < xTrj.length; n++) {
    midX += sx(xTrj[n][0]);
    midY += sy(xTrj[n][1]);
  }
  midX /= xTrj.length;
  midY /= xTrj.length;

  appendEndpoint(
    plotLayer, x0px, y0px, separated ? "start" : "start / end", "start", midX, midY
  );
  if (separated) appendEndpoint(plotLayer, x1px, y1px, "end", "end", midX, midY);

  svg.appendChild(plotLayer);
  host.appendChild(svg);
}

function carPolygon(
  x: Vec,
  sx: (v: number) => number,
  sy: (v: number) => number
): SVGElement {
  const heading = x[2];
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  const corners = [
    [-CAR_W / 2, -CAR_H / 2],
    [CAR_W / 2, -CAR_H / 2],
    [CAR_W / 2, CAR_H / 2],
    [-CAR_W / 2, CAR_H / 2],
  ];
  const pts = corners
    .map(function (p) {
      const wx = x[0] + c * p[0] - s * p[1];
      const wy = x[1] + s * p[0] + c * p[1];
      return sx(wx) + "," + sy(wy);
    })
    .join(" ");
  return svgEl("polygon", { points: pts, class: "car-body" });
}

/** The label is offset away from (midX, midY), the centre of the drawn path. */
function appendEndpoint(
  parent: SVGElement,
  px: number,
  py: number,
  label: string,
  kind: string,
  midX: number,
  midY: number
): void {
  parent.appendChild(
    svgEl("circle", { cx: px, cy: py, r: 5, class: "traj-dot " + kind })
  );

  let dx = px - midX;
  let dy = py - midY;
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) {
    dx = -1;
    dy = 1;
  } else {
    dx /= len;
    dy /= len;
  }

  const text = svgEl("text", {
    x: px + dx * 13,
    y: py + dy * 13 + 4,
    class: "direct-label",
    "text-anchor": dx < -0.2 ? "end" : dx > 0.2 ? "start" : "middle",
  });
  text.textContent = label;
  parent.appendChild(text);
}

/* --------------------------------------------------------------------------
 * Line charts (total cost, and one per control component)
 * ----------------------------------------------------------------------- */

interface LineChartOptions {
  values: number[];
  xDomain: [number, number];
  /** Fixed y-range; omit to fit the values. */
  yDomain?: [number, number];
  /** Persistent marker + crosshair, e.g. the slider's iteration. */
  pointerIndex?: number;
  xAxisLabel: string;
  /** Draw a rule at y = 0 (control plots swing either side of zero). */
  zeroLine?: boolean;
  format: (v: number) => string;
  tooltip: (i: number) => { title: string; rows: string[][] };
  onSeek?: (i: number) => void;
  ariaLabel: string;
  width: number;
  height: number;
}

function renderLineChart(host: HTMLElement, o: LineChartOptions): void {
  clearNode(host);
  const margin = { top: 12, right: 44, bottom: 30, left: 54 };
  const plotW = Math.max(10, o.width - margin.left - margin.right);
  const plotH = Math.max(10, o.height - margin.top - margin.bottom);

  const [xMin, xMax] = o.xDomain;
  const xSpan = xMax - xMin;

  let yLo: number;
  let yHi: number;
  if (o.yDomain) {
    yLo = o.yDomain[0];
    yHi = o.yDomain[1];
  } else {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < o.values.length; i++) {
      lo = Math.min(lo, o.values[i]);
      hi = Math.max(hi, o.values[i]);
    }
    if (hi - lo < 1e-9) hi = lo + 1;
    const pad = (hi - lo) * 0.08;
    yLo = lo - pad;
    yHi = hi + pad;
  }
  if (yHi - yLo < 1e-12) yHi = yLo + 1;

  const sx = (i: number) =>
    xSpan < 1e-12
      ? margin.left + plotW / 2
      : margin.left + ((i - xMin) / xSpan) * plotW;
  const sy = (v: number) =>
    margin.top + plotH - ((v - yLo) / (yHi - yLo)) * plotH;

  const svg = svgEl("svg", {
    width: o.width,
    height: o.height,
    viewBox: `0 0 ${o.width} ${o.height}`,
    role: "img",
    "aria-label": o.ariaLabel,
  });

  const grid = svgEl("g", { class: "grid" });
  const yTicks = niceTicks(yLo, yHi, Math.max(3, Math.floor(plotH / 38)));
  for (let i = 0; i < yTicks.length; i++) {
    const py = sy(yTicks[i]);
    grid.appendChild(
      svgEl("line", { x1: margin.left, y1: py, x2: margin.left + plotW, y2: py })
    );
    const label = svgEl("text", {
      x: margin.left - 7,
      y: py + 4,
      class: "tick",
      "text-anchor": "end",
    });
    label.textContent = o.format(yTicks[i]);
    svg.appendChild(label);
  }
  svg.insertBefore(grid, svg.firstChild);

  const xTicks = integerTicks(xMin, xMax, Math.max(3, Math.floor(plotW / 70)));
  for (let i = 0; i < xTicks.length; i++) {
    const label = svgEl("text", {
      x: sx(xTicks[i]),
      y: margin.top + plotH + 18,
      class: "tick",
      "text-anchor": "middle",
    });
    label.textContent = String(xTicks[i]);
    svg.appendChild(label);
  }

  svg.appendChild(
    svgEl("line", {
      x1: margin.left,
      y1: margin.top + plotH,
      x2: margin.left + plotW,
      y2: margin.top + plotH,
      class: "baseline",
    })
  );

  if (o.zeroLine && yLo < 0 && yHi > 0) {
    svg.appendChild(
      svgEl("line", {
        x1: margin.left,
        y1: sy(0),
        x2: margin.left + plotW,
        y2: sy(0),
        class: "zero-line",
      })
    );
  }

  const xLabel = svgEl("text", {
    x: margin.left + plotW / 2,
    y: o.height - 3,
    class: "axis-title",
    "text-anchor": "middle",
  });
  xLabel.textContent = o.xAxisLabel;
  svg.appendChild(xLabel);

  // The series
  let d = "";
  for (let i = 0; i < o.values.length; i++) {
    d += (i === 0 ? "M" : "L") + sx(i) + " " + sy(o.values[i]);
  }
  svg.appendChild(svgEl("path", { d, class: "cost-line" }));
  if (o.values.length === 1) {
    svg.appendChild(
      svgEl("circle", { cx: sx(0), cy: sy(o.values[0]), r: 4, class: "cost-dot" })
    );
  }

  // Persistent pointer tied to the slider
  if (typeof o.pointerIndex === "number") {
    const idx = Math.max(0, Math.min(o.values.length - 1, o.pointerIndex));
    const px = sx(idx);
    const py = sy(o.values[idx]);
    svg.appendChild(
      svgEl("line", {
        x1: px, y1: margin.top, x2: px, y2: margin.top + plotH, class: "crosshair",
      })
    );
    svg.appendChild(svgEl("circle", { cx: px, cy: py, r: 5, class: "cost-dot" }));

    const flip = px > margin.left + plotW - 46;
    const valueLabel = svgEl("text", {
      x: px + (flip ? -10 : 10),
      y: Math.max(margin.top + 10, py - 10),
      class: "direct-label",
      "text-anchor": flip ? "end" : "start",
    });
    valueLabel.textContent = o.format(o.values[idx]);
    svg.appendChild(valueLabel);
  }

  // Hover layer: crosshair + tooltip, and click-to-seek where supported
  const hoverLine = svgEl("line", {
    x1: 0, y1: margin.top, x2: 0, y2: margin.top + plotH,
    class: "hover-crosshair", visibility: "hidden",
  });
  const hoverDot = svgEl("circle", {
    cx: 0, cy: 0, r: 4, class: "hover-dot", visibility: "hidden",
  });
  svg.appendChild(hoverLine);
  svg.appendChild(hoverDot);

  const overlay = svgEl("rect", {
    x: margin.left, y: margin.top, width: plotW, height: plotH,
    fill: "transparent", cursor: o.onSeek ? "pointer" : "crosshair",
  });
  svg.appendChild(overlay);
  host.appendChild(svg);

  const tip = document.createElement("div");
  tip.className = "chart-tooltip";
  host.appendChild(tip);

  const indexAt = function (clientX: number): number {
    const rect = svg.getBoundingClientRect();
    const scaleX = rect.width / o.width; // the SVG is CSS-scaled to its box
    const local = (clientX - rect.left) / scaleX;
    const frac = plotW > 0 ? (local - margin.left) / plotW : 0;
    const i = Math.round(xMin + frac * xSpan);
    return Math.max(0, Math.min(o.values.length - 1, i));
  };

  overlay.addEventListener("pointermove", function (ev: Event) {
    const i = indexAt((ev as PointerEvent).clientX);
    const px = sx(i);
    const py = sy(o.values[i]);
    hoverLine.setAttribute("x1", String(px));
    hoverLine.setAttribute("x2", String(px));
    hoverLine.setAttribute("visibility", "visible");
    hoverDot.setAttribute("cx", String(px));
    hoverDot.setAttribute("cy", String(py));
    hoverDot.setAttribute("visibility", "visible");

    const t = o.tooltip(i);
    let html = '<div class="tt-title">' + t.title + "</div>";
    for (let r = 0; r < t.rows.length; r++) {
      html +=
        '<div class="tt-row"><span>' + t.rows[r][0] + "</span><b>" +
        t.rows[r][1] + "</b></div>";
    }
    tip.innerHTML = html;
    tip.classList.add("visible");

    const hostRect = host.getBoundingClientRect();
    const localX = (ev as PointerEvent).clientX - hostRect.left;
    let left = localX + 14;
    if (left + tip.offsetWidth > hostRect.width) left = localX - tip.offsetWidth - 14;
    tip.style.left = Math.max(0, left) + "px";
    tip.style.top = "6px";
  });

  overlay.addEventListener("pointerleave", function () {
    hoverLine.setAttribute("visibility", "hidden");
    hoverDot.setAttribute("visibility", "hidden");
    tip.classList.remove("visible");
  });

  if (o.onSeek) {
    const seek = o.onSeek;
    overlay.addEventListener("pointerdown", function (ev: Event) {
      seek(indexAt((ev as PointerEvent).clientX));
    });
  }
}

/**
 * A y-range for one control component, shared by every iteration so the slider
 * shows the controls actually changing rather than the axis rescaling. Early
 * iterations can be wild, so they only count up to a high quantile.
 */
function controlDomain(uTrace: Mat[], comp: number): [number, number] {
  const all: number[] = [];
  for (let i = 0; i < uTrace.length; i++) {
    for (let n = 0; n < uTrace[i].length; n++) all.push(uTrace[i][n][comp]);
  }
  all.sort(function (a, b) { return a - b; });

  let lo = quantile(all, 0.01);
  let hi = quantile(all, 0.99);

  // Always show the converged controls in full
  const final = uTrace[uTrace.length - 1];
  for (let n = 0; n < final.length; n++) {
    lo = Math.min(lo, final[n][comp]);
    hi = Math.max(hi, final[n][comp]);
  }
  if (hi - lo < 1e-6) {
    const mid = (hi + lo) / 2;
    lo = mid - 0.5;
    hi = mid + 0.5;
  }
  const pad = (hi - lo) * 0.12;
  return [lo - pad, hi + pad];
}
