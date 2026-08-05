"use strict";
/**
 * Minimal dense linear algebra on plain arrays.
 * Vectors are number[]; matrices are number[][] in row-major order.
 */
function zeros(n) {
    return new Array(n).fill(0);
}
function zerosMat(rows, cols) {
    const out = [];
    for (let i = 0; i < rows; i++)
        out.push(zeros(cols));
    return out;
}
function eye(n) {
    const out = zerosMat(n, n);
    for (let i = 0; i < n; i++)
        out[i][i] = 1;
    return out;
}
function cloneVec(v) {
    return v.slice();
}
function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++)
        s += a[i] * b[i];
    return s;
}
function addVec(a, b) {
    const out = zeros(a.length);
    for (let i = 0; i < a.length; i++)
        out[i] = a[i] + b[i];
    return out;
}
function subVec(a, b) {
    const out = zeros(a.length);
    for (let i = 0; i < a.length; i++)
        out[i] = a[i] - b[i];
    return out;
}
function scaleVec(a, s) {
    return a.map((v) => v * s);
}
function addMat(A, B) {
    const out = zerosMat(A.length, A[0].length);
    for (let i = 0; i < A.length; i++)
        for (let j = 0; j < A[0].length; j++)
            out[i][j] = A[i][j] + B[i][j];
    return out;
}
function scaleMat(A, s) {
    return A.map((row) => row.map((v) => v * s));
}
function transpose(A) {
    const out = zerosMat(A[0].length, A.length);
    for (let i = 0; i < A.length; i++)
        for (let j = 0; j < A[0].length; j++)
            out[j][i] = A[i][j];
    return out;
}
/** A @ x */
function matVec(A, x) {
    const out = zeros(A.length);
    for (let i = 0; i < A.length; i++) {
        let s = 0;
        for (let j = 0; j < x.length; j++)
            s += A[i][j] * x[j];
        out[i] = s;
    }
    return out;
}
/** A.T @ x */
function matTVec(A, x) {
    const cols = A[0].length;
    const out = zeros(cols);
    for (let j = 0; j < cols; j++) {
        let s = 0;
        for (let i = 0; i < A.length; i++)
            s += A[i][j] * x[i];
        out[j] = s;
    }
    return out;
}
/** A @ B */
function matMul(A, B) {
    const n = A.length;
    const k = B.length;
    const m = B[0].length;
    const out = zerosMat(n, m);
    for (let i = 0; i < n; i++) {
        for (let p = 0; p < k; p++) {
            const a = A[i][p];
            if (a === 0)
                continue;
            for (let j = 0; j < m; j++)
                out[i][j] += a * B[p][j];
        }
    }
    return out;
}
/** A.T @ B */
function matTMul(A, B) {
    return matMul(transpose(A), B);
}
/** Dense inverse via Gauss-Jordan with partial pivoting. */
function inverse(A) {
    const n = A.length;
    const aug = A.map((row, i) => row.concat(eye(n)[i]));
    for (let col = 0; col < n; col++) {
        let pivot = col;
        for (let row = col + 1; row < n; row++) {
            if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col]))
                pivot = row;
        }
        if (Math.abs(aug[pivot][col]) < 1e-14) {
            throw new Error("Matrix is singular and cannot be inverted");
        }
        const tmp = aug[col];
        aug[col] = aug[pivot];
        aug[pivot] = tmp;
        const d = aug[col][col];
        for (let j = 0; j < 2 * n; j++)
            aug[col][j] /= d;
        for (let row = 0; row < n; row++) {
            if (row === col)
                continue;
            const factor = aug[row][col];
            if (factor === 0)
                continue;
            for (let j = 0; j < 2 * n; j++)
                aug[row][j] -= factor * aug[col][j];
        }
    }
    return aug.map((row) => row.slice(n));
}
/** Force exact symmetry; guards against drift in the value-function recursion. */
function symmetrize(A) {
    const n = A.length;
    const out = zerosMat(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            out[i][j] = 0.5 * (A[i][j] + A[j][i]);
    return out;
}
/**
 * Vehicle model and cost functions from ilqr_driving.ipynb.
 *
 * The notebook gets its derivatives from pydrake's symbolic autodiff. There is no
 * autodiff here, so the gradients/Hessians/Jacobians below are the analytic
 * derivatives of the same expressions, written out by hand.
 */
const N_X = 5; // [x position, y position, heading, speed, steering angle]
const N_U = 2; // [acceleration, steering velocity]
const DT = 0.1;
const EPS = 1e-6; // sqrt(x) has undefined derivative at x = 0; smooth it out
const CONTROL_WEIGHT = 0.1;
/**
 * Targets of the cost function. The notebook hard-codes r = 2 and v_target = 2;
 * here they are set from the UI before each solve.
 */
const costParams = { r: 2.0, vTarget: 2.0 };
function setCostTargets(r, vTarget) {
    costParams.r = r;
    costParams.vTarget = vTarget;
}
/** Continuous-time car dynamics: xdot = f_c(x, u). */
function carContinuousDynamics(x, u) {
    const heading = x[2];
    const v = x[3];
    const steer = x[4];
    return [
        v * Math.cos(heading),
        v * Math.sin(heading),
        v * Math.tan(steer),
        u[0],
        u[1],
    ];
}
/** Discrete dynamics via a forward Euler step: x[n+1] = x[n] + dt * f_c(x[n], u[n]). */
function discreteDynamics(x, u) {
    const xd = carContinuousDynamics(x, u);
    const xNext = zeros(N_X);
    for (let i = 0; i < N_X; i++)
        xNext[i] = x[i] + DT * xd[i];
    return xNext;
}
/** Roll the dynamics forward from x0 under u_trj. Returns x_trj of shape [N, n_x]. */
function rollout(x0, uTrj) {
    const xTrj = zerosMat(uTrj.length + 1, x0.length);
    xTrj[0] = cloneVec(x0);
    for (let n = 0; n < uTrj.length; n++) {
        xTrj[n + 1] = discreteDynamics(xTrj[n], uTrj[n]);
    }
    return xTrj;
}
function costStage(x, u) {
    const cCircle = Math.pow(Math.sqrt(x[0] * x[0] + x[1] * x[1] + EPS) - costParams.r, 2);
    const cSpeed = Math.pow(x[3] - costParams.vTarget, 2);
    const cControl = (u[0] * u[0] + u[1] * u[1]) * CONTROL_WEIGHT;
    return cCircle + cSpeed + cControl;
}
function costFinal(x) {
    const cCircle = Math.pow(Math.sqrt(x[0] * x[0] + x[1] * x[1] + EPS) - costParams.r, 2);
    const cSpeed = Math.pow(x[3] - costParams.vTarget, 2);
    return cCircle + cSpeed;
}
/** Total trajectory cost: sum of stage costs plus the final cost. */
function costTrj(xTrj, uTrj) {
    let total = 0.0;
    for (let n = 0; n < uTrj.length; n++) {
        total += costStage(xTrj[n], uTrj[n]);
    }
    total += costFinal(xTrj[xTrj.length - 1]);
    return total;
}
/**
 * Gradient and Hessian of the position term (sqrt(px^2 + py^2 + eps) - r)^2,
 * shared by the stage and final costs.
 */
function circleCostDerivs(px, py) {
    const s = Math.sqrt(px * px + py * py + EPS);
    const s3 = s * s * s;
    return {
        gx: 2 * px * (1 - costParams.r / s),
        gy: 2 * py * (1 - costParams.r / s),
        hxx: 2 * (1 - costParams.r / s) + (2 * costParams.r * px * px) / s3,
        hxy: (2 * costParams.r * px * py) / s3,
        hyy: 2 * (1 - costParams.r / s) + (2 * costParams.r * py * py) / s3,
    };
}
/** Cost derivatives and dynamics Jacobians at a stage (x, u). */
function stageDerivatives(x, u) {
    const heading = x[2];
    const v = x[3];
    const steer = x[4];
    const c = circleCostDerivs(x[0], x[1]);
    const l_x = zeros(N_X);
    l_x[0] = c.gx;
    l_x[1] = c.gy;
    l_x[3] = 2 * (v - costParams.vTarget);
    const l_xx = zerosMat(N_X, N_X);
    l_xx[0][0] = c.hxx;
    l_xx[0][1] = c.hxy;
    l_xx[1][0] = c.hxy;
    l_xx[1][1] = c.hyy;
    l_xx[3][3] = 2;
    const l_u = [2 * CONTROL_WEIGHT * u[0], 2 * CONTROL_WEIGHT * u[1]];
    const l_uu = [
        [2 * CONTROL_WEIGHT, 0],
        [0, 2 * CONTROL_WEIGHT],
    ];
    const l_ux = zerosMat(N_U, N_X); // costs are separable in x and u
    // f_x = I + dt * d(xdot)/dx
    const cosH = Math.cos(heading);
    const sinH = Math.sin(heading);
    const cosS = Math.cos(steer);
    const f_x = eye(N_X);
    f_x[0][2] += DT * (-v * sinH);
    f_x[0][3] += DT * cosH;
    f_x[1][2] += DT * (v * cosH);
    f_x[1][3] += DT * sinH;
    f_x[2][3] += DT * Math.tan(steer);
    f_x[2][4] += DT * (v / (cosS * cosS));
    // f_u = dt * d(xdot)/du
    const f_u = zerosMat(N_X, N_U);
    f_u[3][0] = DT;
    f_u[4][1] = DT;
    return { l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u };
}
/** Gradient and Hessian of the final cost at x. */
function finalDerivatives(x) {
    const c = circleCostDerivs(x[0], x[1]);
    const l_final_x = zeros(N_X);
    l_final_x[0] = c.gx;
    l_final_x[1] = c.gy;
    l_final_x[3] = 2 * (x[3] - costParams.vTarget);
    const l_final_xx = zerosMat(N_X, N_X);
    l_final_xx[0][0] = c.hxx;
    l_final_xx[0][1] = c.hxy;
    l_final_xx[1][0] = c.hxy;
    l_final_xx[1][1] = c.hyy;
    l_final_xx[3][3] = 2;
    return { l_final_x, l_final_xx };
}
/**
 * iLQR: the TODO cells of ilqr_driving.ipynb, filled in.
 */
/**
 * Coefficients of the quadratic expansion of Q(x, u) = l(x, u) + V(f(x, u)),
 * collected from the expansions of l and f around the nominal (xbar, ubar).
 * Uses dx[n+1] = f_x dx[n] + f_u du[n].
 */
function QTermsOf(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx) {
    const Q_x = addVec(l_x, matTVec(f_x, V_x));
    const Q_u = addVec(l_u, matTVec(f_u, V_x));
    const Q_xx = addMat(l_xx, matMul(matTMul(f_x, V_xx), f_x));
    const Q_ux = addMat(l_ux, matMul(matTMul(f_u, V_xx), f_x));
    const Q_uu = addMat(l_uu, matMul(matTMul(f_u, V_xx), f_u));
    return { Q_x, Q_u, Q_xx, Q_ux, Q_uu };
}
/**
 * Minimiser of the quadratic Q in du: du* = k + K dx, from dQ/ddu = 0
 * => Q_u + Q_uu du + Q_ux dx = 0.
 */
function gains(Q_uu, Q_u, Q_ux) {
    const Q_uu_inv = inverse(Q_uu);
    const k = scaleVec(matVec(Q_uu_inv, Q_u), -1);
    const K = scaleMat(matMul(Q_uu_inv, Q_ux), -1);
    return { k, K };
}
/**
 * Value function at n, obtained by substituting du* = k + K dx into Q and
 * collecting the terms in dx and 1/2 dx' (.) dx. Written for general k and K
 * (i.e. without assuming they come from `gains`).
 */
function VTerms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k) {
    // V_x = Q_x + K' Q_u + Q_ux' k + K' Q_uu k
    let V_x = addVec(Q_x, matTVec(K, Q_u));
    V_x = addVec(V_x, matTVec(Q_ux, k));
    V_x = addVec(V_x, matTVec(K, matVec(Q_uu, k)));
    // V_xx = Q_xx + K' Q_ux + Q_ux' K + K' Q_uu K
    const KtQux = matTMul(K, Q_ux);
    let V_xx = addMat(Q_xx, KtQux);
    V_xx = addMat(V_xx, transpose(KtQux));
    V_xx = addMat(V_xx, matMul(matTMul(K, Q_uu), K));
    return { V_x, V_xx: symmetrize(V_xx) };
}
/** Cost drop predicted by the quadratic model when du = k. */
function expectedCostReduction(Q_u, Q_uu, k) {
    return -dot(Q_u, k) - 0.5 * dot(k, matVec(Q_uu, k));
}
/**
 * Apply the affine feedback law along the trajectory and propagate the true
 * dynamics: u' = ubar + k + K (x' - xbar).
 */
function forwardPass(xTrj, uTrj, kTrj, KTrj) {
    const xTrjNew = zerosMat(xTrj.length, xTrj[0].length);
    xTrjNew[0] = cloneVec(xTrj[0]);
    const uTrjNew = zerosMat(uTrj.length, uTrj[0].length);
    for (let n = 0; n < uTrj.length; n++) {
        const dx = subVec(xTrjNew[n], xTrj[n]);
        uTrjNew[n] = addVec(uTrj[n], addVec(kTrj[n], matVec(KTrj[n], dx)));
        xTrjNew[n + 1] = discreteDynamics(xTrjNew[n], uTrjNew[n]);
    }
    return { xTrjNew, uTrjNew };
}
/**
 * Backward Riccati-like recursion from the terminal condition
 * V(x[N]) = l_f(x[N]) down to n = 0, producing the gain trajectories.
 */
function backwardPass(xTrj, uTrj, regu) {
    const nU = uTrj[0].length;
    const kTrj = zerosMat(uTrj.length, nU);
    const KTrj = [];
    for (let n = 0; n < uTrj.length; n++)
        KTrj.push(zerosMat(nU, xTrj[0].length));
    let expectedCostRedu = 0;
    // Terminal boundary condition
    const fin = finalDerivatives(xTrj[xTrj.length - 1]);
    let V_x = fin.l_final_x;
    let V_xx = fin.l_final_xx;
    for (let n = uTrj.length - 1; n >= 0; n--) {
        const d = stageDerivatives(xTrj[n], uTrj[n]);
        const q = QTermsOf(d.l_x, d.l_u, d.l_xx, d.l_ux, d.l_uu, d.f_x, d.f_u, V_x, V_xx);
        // Regularization keeps Q_uu invertible and well conditioned; it acts as a
        // quadratic penalty on stepping away from the previous control trajectory.
        const Q_uu_regu = addMat(q.Q_uu, scaleMat(eye(nU), regu));
        const g = gains(Q_uu_regu, q.Q_u, q.Q_ux);
        kTrj[n] = g.k;
        KTrj[n] = g.K;
        const v = VTerms(q.Q_x, q.Q_u, q.Q_xx, q.Q_ux, q.Q_uu, g.K, g.k);
        V_x = v.V_x;
        V_xx = v.V_xx;
        expectedCostRedu += expectedCostReduction(q.Q_u, q.Q_uu, g.k);
    }
    return { kTrj, KTrj, expectedCostRedu };
}
/** Deterministic PRNG so a given seed always reproduces the same run. */
function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
function makeGaussian(seed) {
    const rand = mulberry32(seed);
    return function () {
        // Box-Muller
        const u1 = Math.max(rand(), 1e-12);
        const u2 = rand();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
}
/**
 * Main loop: alternate backward and forward passes, accepting the new pair and
 * relaxing the regularization when the cost drops, rejecting and stiffening it
 * when it does not.
 */
function runIlqr(x0, N, maxIter, reguInit, seed) {
    const randn = makeGaussian(seed);
    // First forward rollout from a near-zero control guess
    let uTrj = zerosMat(N - 1, N_U);
    for (let n = 0; n < uTrj.length; n++)
        for (let i = 0; i < N_U; i++)
            uTrj[n][i] = randn() * 0.0001;
    let xTrj = rollout(x0, uTrj);
    let totalCost = costTrj(xTrj, uTrj);
    let regu = reguInit;
    const maxRegu = 10000;
    const minRegu = 0.01;
    const costTrace = [totalCost];
    const reduRatioTrace = [1];
    const reduTrace = [];
    const reguTrace = [regu];
    const xTrace = [xTrj];
    const uTrace = [uTrj];
    let converged = false;
    let iterationsRun = 0;
    for (let it = 0; it < maxIter; it++) {
        iterationsRun = it + 1;
        const bp = backwardPass(xTrj, uTrj, regu);
        const fp = forwardPass(xTrj, uTrj, bp.kTrj, bp.KTrj);
        totalCost = costTrj(fp.xTrjNew, fp.uTrjNew);
        const costRedu = costTrace[costTrace.length - 1] - totalCost;
        const reduRatio = costRedu / Math.abs(bp.expectedCostRedu);
        if (costRedu > 0) {
            // Improvement: accept the new trajectories and lower the regularization
            reduRatioTrace.push(reduRatio);
            costTrace.push(totalCost);
            xTrj = fp.xTrjNew;
            uTrj = fp.uTrjNew;
            regu *= 0.7;
        }
        else {
            // Reject and increase the regularization
            regu *= 2.0;
            costTrace.push(costTrace[costTrace.length - 1]);
            reduRatioTrace.push(0);
        }
        xTrace.push(xTrj);
        uTrace.push(uTrj);
        regu = Math.min(Math.max(regu, minRegu), maxRegu);
        reguTrace.push(regu);
        reduTrace.push(costRedu);
        // Early termination if the expected improvement is small
        if (bp.expectedCostRedu <= 1e-6) {
            converged = true;
            break;
        }
    }
    return {
        xTrace,
        uTrace,
        costTrace,
        reguTrace,
        reduRatioTrace,
        reduTrace,
        converged,
        iterationsRun,
    };
}
/**
 * SVG rendering for the four plots. No chart library: everything is drawn from
 * the trajectory, control and cost traces directly.
 */
const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(name, attrs) {
    const el = document.createElementNS(SVG_NS, name);
    for (const key in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, key)) {
            el.setAttribute(key, String(attrs[key]));
        }
    }
    return el;
}
function clearNode(node) {
    while (node.firstChild)
        node.removeChild(node.firstChild);
}
/** Tick positions on a 1-2-5 ladder covering [min, max]. */
function niceTicks(min, max, target) {
    if (!isFinite(min) || !isFinite(max))
        return [0];
    if (max - min < 1e-12)
        return [min];
    const rawStep = (max - min) / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    let step;
    if (norm <= 1)
        step = 1;
    else if (norm <= 2)
        step = 2;
    else if (norm <= 5)
        step = 5;
    else
        step = 10;
    step *= mag;
    const ticks = [];
    const start = Math.ceil(min / step) * step;
    for (let v = start; v <= max + step * 1e-6; v += step) {
        ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return ticks;
}
/** Integer tick positions for the iteration / timestep axes. */
function integerTicks(min, max, target) {
    const span = Math.max(1, max - min);
    let step = Math.max(1, Math.ceil(span / target));
    const mag = Math.pow(10, Math.floor(Math.log10(step)));
    const norm = step / mag;
    if (norm <= 1)
        step = mag;
    else if (norm <= 2)
        step = 2 * mag;
    else if (norm <= 5)
        step = 5 * mag;
    else
        step = 10 * mag;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max; v += step)
        ticks.push(v);
    if (ticks.length === 0 || ticks[ticks.length - 1] !== max)
        ticks.push(max);
    return ticks;
}
function formatCost(v) {
    const a = Math.abs(v);
    if (a === 0)
        return "0";
    if (a >= 1000)
        return v.toFixed(0);
    if (a >= 100)
        return v.toFixed(1);
    if (a >= 1)
        return v.toFixed(2);
    return v.toPrecision(3);
}
/** Signed, fixed-width-ish formatting for the control values. */
function formatControl(v) {
    const a = Math.abs(v);
    if (a === 0)
        return "0";
    if (a >= 100)
        return v.toFixed(0);
    if (a >= 1)
        return v.toFixed(2);
    return v.toFixed(3);
}
function quantile(sorted, q) {
    if (sorted.length === 0)
        return 0;
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
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
function trajectoryWindow(xTrace) {
    const r = costParams.r;
    let xMin = -r;
    let xMax = r;
    let yMin = -r;
    let yMax = r;
    // Must-show: the initial state and the converged trajectory
    const mustShow = [xTrace[0][0]].concat(xTrace[xTrace.length - 1]);
    for (let n = 0; n < mustShow.length; n++) {
        xMin = Math.min(xMin, mustShow[n][0]);
        xMax = Math.max(xMax, mustShow[n][0]);
        yMin = Math.min(yMin, mustShow[n][1]);
        yMax = Math.max(yMax, mustShow[n][1]);
    }
    // Nice-to-show: the bulk of every intermediate iteration
    const xs = [];
    const ys = [];
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
function renderTrajectory(host, xTrj, win, width, height) {
    clearNode(host);
    const margin = { top: 12, right: 14, bottom: 26, left: 40 };
    const plotW = Math.max(10, width - margin.left - margin.right);
    const plotH = Math.max(10, height - margin.top - margin.bottom);
    // Equal aspect: one scale for both axes, centred in the plot area
    const scale = Math.min(plotW / (win.xMax - win.xMin), plotH / (win.yMax - win.yMin));
    const cx = (win.xMin + win.xMax) / 2;
    const cy = (win.yMin + win.yMax) / 2;
    const originX = margin.left + plotW / 2;
    const originY = margin.top + plotH / 2;
    const sx = (x) => originX + (x - cx) * scale;
    const sy = (y) => originY - (y - cy) * scale;
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
        grid.appendChild(svgEl("line", { x1: px, y1: margin.top, x2: px, y2: margin.top + plotH }));
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
        grid.appendChild(svgEl("line", { x1: margin.left, y1: py, x2: margin.left + plotW, y2: py }));
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
    clip.appendChild(svgEl("rect", { x: margin.left, y: margin.top, width: plotW, height: plotH }));
    defs.appendChild(clip);
    svg.appendChild(defs);
    const plotLayer = svgEl("g", { "clip-path": "url(#" + clipId + ")" });
    // Target circle (the cost pulls the car onto this)
    plotLayer.appendChild(svgEl("circle", {
        cx: sx(0),
        cy: sy(0),
        r: costParams.r * scale,
        class: "target-circle",
    }));
    // Car bodies, thinned out so they stay readable at long horizons and do not
    // pile into a solid block where the car is moving slowly.
    const bodies = svgEl("g", { class: "car-bodies" });
    const step = Math.max(1, Math.round(xTrj.length / 26));
    let lastX = NaN;
    let lastY = NaN;
    for (let n = 0; n < xTrj.length; n += step) {
        const px = sx(xTrj[n][0]);
        const py = sy(xTrj[n][1]);
        if (n > 0 && Math.hypot(px - lastX, py - lastY) < 14)
            continue;
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
        knots.appendChild(svgEl("circle", { cx: sx(xTrj[n][0]), cy: sy(xTrj[n][1]), r: 2.4 }));
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
    appendEndpoint(plotLayer, x0px, y0px, separated ? "start" : "start / end", "start", midX, midY);
    if (separated)
        appendEndpoint(plotLayer, x1px, y1px, "end", "end", midX, midY);
    svg.appendChild(plotLayer);
    host.appendChild(svg);
}
function carPolygon(x, sx, sy) {
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
function appendEndpoint(parent, px, py, label, kind, midX, midY) {
    parent.appendChild(svgEl("circle", { cx: px, cy: py, r: 5, class: "traj-dot " + kind }));
    let dx = px - midX;
    let dy = py - midY;
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) {
        dx = -1;
        dy = 1;
    }
    else {
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
function renderLineChart(host, o) {
    clearNode(host);
    const margin = { top: 12, right: 44, bottom: 30, left: 54 };
    const plotW = Math.max(10, o.width - margin.left - margin.right);
    const plotH = Math.max(10, o.height - margin.top - margin.bottom);
    const [xMin, xMax] = o.xDomain;
    const xSpan = xMax - xMin;
    let yLo;
    let yHi;
    if (o.yDomain) {
        yLo = o.yDomain[0];
        yHi = o.yDomain[1];
    }
    else {
        let lo = Infinity;
        let hi = -Infinity;
        for (let i = 0; i < o.values.length; i++) {
            lo = Math.min(lo, o.values[i]);
            hi = Math.max(hi, o.values[i]);
        }
        if (hi - lo < 1e-9)
            hi = lo + 1;
        const pad = (hi - lo) * 0.08;
        yLo = lo - pad;
        yHi = hi + pad;
    }
    if (yHi - yLo < 1e-12)
        yHi = yLo + 1;
    const sx = (i) => xSpan < 1e-12
        ? margin.left + plotW / 2
        : margin.left + ((i - xMin) / xSpan) * plotW;
    const sy = (v) => margin.top + plotH - ((v - yLo) / (yHi - yLo)) * plotH;
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
        grid.appendChild(svgEl("line", { x1: margin.left, y1: py, x2: margin.left + plotW, y2: py }));
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
    svg.appendChild(svgEl("line", {
        x1: margin.left,
        y1: margin.top + plotH,
        x2: margin.left + plotW,
        y2: margin.top + plotH,
        class: "baseline",
    }));
    if (o.zeroLine && yLo < 0 && yHi > 0) {
        svg.appendChild(svgEl("line", {
            x1: margin.left,
            y1: sy(0),
            x2: margin.left + plotW,
            y2: sy(0),
            class: "zero-line",
        }));
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
        svg.appendChild(svgEl("circle", { cx: sx(0), cy: sy(o.values[0]), r: 4, class: "cost-dot" }));
    }
    // Persistent pointer tied to the slider
    if (typeof o.pointerIndex === "number") {
        const idx = Math.max(0, Math.min(o.values.length - 1, o.pointerIndex));
        const px = sx(idx);
        const py = sy(o.values[idx]);
        svg.appendChild(svgEl("line", {
            x1: px, y1: margin.top, x2: px, y2: margin.top + plotH, class: "crosshair",
        }));
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
    const indexAt = function (clientX) {
        const rect = svg.getBoundingClientRect();
        const scaleX = rect.width / o.width; // the SVG is CSS-scaled to its box
        const local = (clientX - rect.left) / scaleX;
        const frac = plotW > 0 ? (local - margin.left) / plotW : 0;
        const i = Math.round(xMin + frac * xSpan);
        return Math.max(0, Math.min(o.values.length - 1, i));
    };
    overlay.addEventListener("pointermove", function (ev) {
        const i = indexAt(ev.clientX);
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
        const localX = ev.clientX - hostRect.left;
        let left = localX + 14;
        if (left + tip.offsetWidth > hostRect.width)
            left = localX - tip.offsetWidth - 14;
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
        overlay.addEventListener("pointerdown", function (ev) {
            seek(indexAt(ev.clientX));
        });
    }
}
/**
 * A y-range for one control component, shared by every iteration so the slider
 * shows the controls actually changing rather than the axis rescaling. Early
 * iterations can be wild, so they only count up to a high quantile.
 */
function controlDomain(uTrace, comp) {
    const all = [];
    for (let i = 0; i < uTrace.length; i++) {
        for (let n = 0; n < uTrace[i].length; n++)
            all.push(uTrace[i][n][comp]);
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
/**
 * UI wiring: read the problem setup, run iLQR, and scrub through iterations.
 */
const state = {
    result: null,
    win: null,
    accelDomain: null,
    steerDomain: null,
    iter: 0,
};
function el(id) {
    const node = document.getElementById(id);
    if (!node)
        throw new Error("Missing element #" + id);
    return node;
}
const x0Input = el("x0-input");
const radiusInput = el("radius-input");
const speedInput = el("speed-input");
const horizonInput = el("horizon-input");
const maxIterInput = el("maxiter-input");
const reguInput = el("regu-input");
const runButton = el("run-button");
const errorBox = el("error-box");
const statusBox = el("status-box");
const slider = el("iter-slider");
const sliderReadout = el("iter-readout");
const sliderMaxLabel = el("slider-max");
const trajHost = el("traj-plot");
const costHost = el("cost-plot");
const accelHost = el("accel-plot");
const steerHost = el("steer-plot");
const dataButton = el("data-button");
const dataPanel = el("data-panel");
const panelClose = el("panel-close");
const mathButton = el("math-button");
const mathPanel = el("math-panel");
const mathClose = el("math-close");
const mathParams = el("math-params");
const panelSub = el("panel-sub");
const traceTableBody = el("trace-table-body");
const trajTableBody = el("traj-table-body");
const trajTableTitle = el("traj-table-title");
/** Parse "[1, 0, 0, 1, 0]" (brackets and separators are all optional). */
function parseState(text) {
    const cleaned = text.replace(/[\[\]()]/g, " ").trim();
    const parts = cleaned.split(/[\s,;]+/).filter(function (p) {
        return p.length > 0;
    });
    if (parts.length !== N_X) {
        throw new Error("Initial state needs " + N_X + " numbers [x, y, heading, speed, steering]" +
            ", got " + parts.length);
    }
    return parts.map(function (p) {
        const v = Number(p);
        if (!isFinite(v))
            throw new Error('"' + p + '" is not a number');
        return v;
    });
}
function parseInt_(input, name, min) {
    const v = Number(input.value);
    if (!isFinite(v) || Math.floor(v) !== v || v < min) {
        throw new Error(name + " must be an integer >= " + min);
    }
    return v;
}
function parseNumber(input, name, min, allowMin) {
    const v = Number(input.value);
    const ok = isFinite(v) && (allowMin ? v >= min : v > min);
    if (input.value.trim() === "" || !ok) {
        throw new Error(name + " must be a number " + (allowMin ? ">= " : "> ") + min);
    }
    return v;
}
function run() {
    errorBox.textContent = "";
    errorBox.classList.remove("visible");
    let x0;
    let N;
    let maxIter;
    let reguInit;
    let radius;
    let speed;
    try {
        x0 = parseState(x0Input.value);
        radius = parseNumber(radiusInput, "Target radius", 0, true);
        speed = parseNumber(speedInput, "Target speed", -Infinity, true);
        N = parseInt_(horizonInput, "Horizon N", 2);
        maxIter = parseInt_(maxIterInput, "Max iterations", 1);
        reguInit = parseNumber(reguInput, "Initial regularization", 0, false);
    }
    catch (e) {
        errorBox.textContent = e.message;
        errorBox.classList.add("visible");
        return;
    }
    setCostTargets(radius, speed);
    let result;
    const t0 = performance.now();
    try {
        result = runIlqr(x0, N, maxIter, reguInit, 12345);
    }
    catch (e) {
        errorBox.textContent = "iLQR failed: " + e.message;
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
function addRow(body, cells) {
    const tr = document.createElement("tr");
    for (let c = 0; c < cells.length; c++) {
        const td = document.createElement("td");
        td.textContent = cells[c];
        tr.appendChild(td);
    }
    body.appendChild(tr);
}
function fillTraceTable(result) {
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
function fillTrajTable(result, iter) {
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
function seek(i) {
    state.iter = i;
    slider.value = String(i);
    render();
}
function render() {
    const result = state.result;
    if (!result || !state.win)
        return;
    const iter = Math.min(state.iter, result.xTrace.length - 1);
    const lastIndex = result.costTrace.length - 1;
    sliderReadout.textContent = String(iter);
    renderTrajectory(trajHost, result.xTrace[iter], state.win, trajHost.clientWidth, trajHost.clientHeight);
    renderLineChart(costHost, {
        values: result.costTrace,
        xDomain: [0, lastIndex],
        pointerIndex: iter,
        xAxisLabel: "iteration",
        format: formatCost,
        ariaLabel: "Total cost per iLQR iteration",
        tooltip: function (i) {
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
    const controls = [
        { host: accelHost, comp: 0, domain: state.accelDomain, name: "Acceleration", unit: "m/s²" },
        { host: steerHost, comp: 1, domain: state.steerDomain, name: "Steering velocity", unit: "rad/s" },
    ];
    for (let c = 0; c < controls.length; c++) {
        const spec = controls[c];
        const values = [];
        for (let n = 0; n < uTrj.length; n++)
            values.push(uTrj[n][spec.comp]);
        renderLineChart(spec.host, {
            values,
            xDomain: [0, Math.max(1, values.length - 1)],
            yDomain: spec.domain,
            xAxisLabel: "timestep n",
            zeroLine: true,
            format: formatControl,
            ariaLabel: spec.name + " control trajectory at iteration " + iter,
            tooltip: function (i) {
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
[x0Input, radiusInput, speedInput, horizonInput, maxIterInput, reguInput].forEach(function (input) {
    input.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter")
            run();
    });
});
/* Data panel ---------------------------------------------------------- */
function openPanel() {
    const result = state.result;
    if (!result)
        return;
    dataPanel.classList.add("visible");
    fillTrajTable(result, state.iter);
    panelSub.textContent =
        "iteration " + state.iter + " of " + (result.costTrace.length - 1);
}
function closePanel() {
    dataPanel.classList.remove("visible");
}
dataButton.addEventListener("click", openPanel);
panelClose.addEventListener("click", closePanel);
dataPanel.addEventListener("pointerdown", function (ev) {
    if (ev.target === dataPanel)
        closePanel();
});
/* Math panel ---------------------------------------------------------- */
/** The symbols in the formulas, filled in with the values of the current run. */
function fillMathParams() {
    const rows = [
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
function revealMath() {
    const scroll = document.querySelector(".math-scroll");
    if (scroll)
        scroll.classList.add("ready");
}
const mathJax = window.MathJax;
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
mathPanel.addEventListener("pointerdown", function (ev) {
    if (ev.target === mathPanel)
        mathPanel.classList.remove("visible");
});
document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape") {
        closePanel();
        mathPanel.classList.remove("visible");
    }
});
/* Keep the SVGs matched to their boxes ------------------------------- */
let resizeFrame = 0;
window.addEventListener("resize", function () {
    if (resizeFrame)
        cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(function () {
        resizeFrame = 0;
        render();
    });
});
// Run once with the notebook's own setup so the page is not empty on load.
run();
//# sourceMappingURL=app.js.map