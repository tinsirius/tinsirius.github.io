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

function setCostTargets(r: number, vTarget: number): void {
  costParams.r = r;
  costParams.vTarget = vTarget;
}

/** Continuous-time car dynamics: xdot = f_c(x, u). */
function carContinuousDynamics(x: Vec, u: Vec): Vec {
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
function discreteDynamics(x: Vec, u: Vec): Vec {
  const xd = carContinuousDynamics(x, u);
  const xNext = zeros(N_X);
  for (let i = 0; i < N_X; i++) xNext[i] = x[i] + DT * xd[i];
  return xNext;
}

/** Roll the dynamics forward from x0 under u_trj. Returns x_trj of shape [N, n_x]. */
function rollout(x0: Vec, uTrj: Mat): Mat {
  const xTrj: Mat = zerosMat(uTrj.length + 1, x0.length);
  xTrj[0] = cloneVec(x0);
  for (let n = 0; n < uTrj.length; n++) {
    xTrj[n + 1] = discreteDynamics(xTrj[n], uTrj[n]);
  }
  return xTrj;
}

function costStage(x: Vec, u: Vec): number {
  const cCircle =
    Math.pow(Math.sqrt(x[0] * x[0] + x[1] * x[1] + EPS) - costParams.r, 2);
  const cSpeed = Math.pow(x[3] - costParams.vTarget, 2);
  const cControl = (u[0] * u[0] + u[1] * u[1]) * CONTROL_WEIGHT;
  return cCircle + cSpeed + cControl;
}

function costFinal(x: Vec): number {
  const cCircle =
    Math.pow(Math.sqrt(x[0] * x[0] + x[1] * x[1] + EPS) - costParams.r, 2);
  const cSpeed = Math.pow(x[3] - costParams.vTarget, 2);
  return cCircle + cSpeed;
}

/** Total trajectory cost: sum of stage costs plus the final cost. */
function costTrj(xTrj: Mat, uTrj: Mat): number {
  let total = 0.0;
  for (let n = 0; n < uTrj.length; n++) {
    total += costStage(xTrj[n], uTrj[n]);
  }
  total += costFinal(xTrj[xTrj.length - 1]);
  return total;
}

interface StageDerivatives {
  l_x: Vec;
  l_u: Vec;
  l_xx: Mat;
  l_ux: Mat;
  l_uu: Mat;
  f_x: Mat;
  f_u: Mat;
}

interface FinalDerivatives {
  l_final_x: Vec;
  l_final_xx: Mat;
}

/**
 * Gradient and Hessian of the position term (sqrt(px^2 + py^2 + eps) - r)^2,
 * shared by the stage and final costs.
 */
function circleCostDerivs(px: number, py: number): {
  gx: number;
  gy: number;
  hxx: number;
  hxy: number;
  hyy: number;
} {
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
function stageDerivatives(x: Vec, u: Vec): StageDerivatives {
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
function finalDerivatives(x: Vec): FinalDerivatives {
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
