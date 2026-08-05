/**
 * iLQR: the TODO cells of ilqr_driving.ipynb, filled in.
 */

interface QTerms {
  Q_x: Vec;
  Q_u: Vec;
  Q_xx: Mat;
  Q_ux: Mat;
  Q_uu: Mat;
}

/**
 * Coefficients of the quadratic expansion of Q(x, u) = l(x, u) + V(f(x, u)),
 * collected from the expansions of l and f around the nominal (xbar, ubar).
 * Uses dx[n+1] = f_x dx[n] + f_u du[n].
 */
function QTermsOf(
  l_x: Vec,
  l_u: Vec,
  l_xx: Mat,
  l_ux: Mat,
  l_uu: Mat,
  f_x: Mat,
  f_u: Mat,
  V_x: Vec,
  V_xx: Mat
): QTerms {
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
function gains(Q_uu: Mat, Q_u: Vec, Q_ux: Mat): { k: Vec; K: Mat } {
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
function VTerms(
  Q_x: Vec,
  Q_u: Vec,
  Q_xx: Mat,
  Q_ux: Mat,
  Q_uu: Mat,
  K: Mat,
  k: Vec
): { V_x: Vec; V_xx: Mat } {
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
function expectedCostReduction(Q_u: Vec, Q_uu: Mat, k: Vec): number {
  return -dot(Q_u, k) - 0.5 * dot(k, matVec(Q_uu, k));
}

/**
 * Apply the affine feedback law along the trajectory and propagate the true
 * dynamics: u' = ubar + k + K (x' - xbar).
 */
function forwardPass(
  xTrj: Mat,
  uTrj: Mat,
  kTrj: Mat,
  KTrj: Mat[]
): { xTrjNew: Mat; uTrjNew: Mat } {
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
function backwardPass(
  xTrj: Mat,
  uTrj: Mat,
  regu: number
): { kTrj: Mat; KTrj: Mat[]; expectedCostRedu: number } {
  const nU = uTrj[0].length;
  const kTrj = zerosMat(uTrj.length, nU);
  const KTrj: Mat[] = [];
  for (let n = 0; n < uTrj.length; n++) KTrj.push(zerosMat(nU, xTrj[0].length));
  let expectedCostRedu = 0;

  // Terminal boundary condition
  const fin = finalDerivatives(xTrj[xTrj.length - 1]);
  let V_x = fin.l_final_x;
  let V_xx = fin.l_final_xx;

  for (let n = uTrj.length - 1; n >= 0; n--) {
    const d = stageDerivatives(xTrj[n], uTrj[n]);
    const q = QTermsOf(
      d.l_x, d.l_u, d.l_xx, d.l_ux, d.l_uu, d.f_x, d.f_u, V_x, V_xx
    );

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

interface IlqrResult {
  /** State trajectory after each iteration; index 0 is the initial rollout. */
  xTrace: Mat[];
  /** Control trajectory after each iteration; index 0 is the initial guess. */
  uTrace: Mat[];
  costTrace: number[];
  reguTrace: number[];
  reduRatioTrace: number[];
  reduTrace: number[];
  /** True if the loop stopped on the expected-reduction threshold. */
  converged: boolean;
  iterationsRun: number;
}

/** Deterministic PRNG so a given seed always reproduces the same run. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGaussian(seed: number): () => number {
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
function runIlqr(
  x0: Vec,
  N: number,
  maxIter: number,
  reguInit: number,
  seed: number
): IlqrResult {
  const randn = makeGaussian(seed);

  // First forward rollout from a near-zero control guess
  let uTrj = zerosMat(N - 1, N_U);
  for (let n = 0; n < uTrj.length; n++)
    for (let i = 0; i < N_U; i++) uTrj[n][i] = randn() * 0.0001;
  let xTrj = rollout(x0, uTrj);
  let totalCost = costTrj(xTrj, uTrj);

  let regu = reguInit;
  const maxRegu = 10000;
  const minRegu = 0.01;

  const costTrace = [totalCost];
  const reduRatioTrace = [1];
  const reduTrace: number[] = [];
  const reguTrace = [regu];
  const xTrace: Mat[] = [xTrj];
  const uTrace: Mat[] = [uTrj];

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
    } else {
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
