/**
 * Minimal dense linear algebra on plain arrays.
 * Vectors are number[]; matrices are number[][] in row-major order.
 */

type Vec = number[];
type Mat = number[][];

function zeros(n: number): Vec {
  return new Array<number>(n).fill(0);
}

function zerosMat(rows: number, cols: number): Mat {
  const out: Mat = [];
  for (let i = 0; i < rows; i++) out.push(zeros(cols));
  return out;
}

function eye(n: number): Mat {
  const out = zerosMat(n, n);
  for (let i = 0; i < n; i++) out[i][i] = 1;
  return out;
}

function cloneVec(v: Vec): Vec {
  return v.slice();
}

function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function addVec(a: Vec, b: Vec): Vec {
  const out = zeros(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function subVec(a: Vec, b: Vec): Vec {
  const out = zeros(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

function scaleVec(a: Vec, s: number): Vec {
  return a.map((v) => v * s);
}

function addMat(A: Mat, B: Mat): Mat {
  const out = zerosMat(A.length, A[0].length);
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[0].length; j++) out[i][j] = A[i][j] + B[i][j];
  return out;
}

function scaleMat(A: Mat, s: number): Mat {
  return A.map((row) => row.map((v) => v * s));
}

function transpose(A: Mat): Mat {
  const out = zerosMat(A[0].length, A.length);
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[0].length; j++) out[j][i] = A[i][j];
  return out;
}

/** A @ x */
function matVec(A: Mat, x: Vec): Vec {
  const out = zeros(A.length);
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    for (let j = 0; j < x.length; j++) s += A[i][j] * x[j];
    out[i] = s;
  }
  return out;
}

/** A.T @ x */
function matTVec(A: Mat, x: Vec): Vec {
  const cols = A[0].length;
  const out = zeros(cols);
  for (let j = 0; j < cols; j++) {
    let s = 0;
    for (let i = 0; i < A.length; i++) s += A[i][j] * x[i];
    out[j] = s;
  }
  return out;
}

/** A @ B */
function matMul(A: Mat, B: Mat): Mat {
  const n = A.length;
  const k = B.length;
  const m = B[0].length;
  const out = zerosMat(n, m);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) out[i][j] += a * B[p][j];
    }
  }
  return out;
}

/** A.T @ B */
function matTMul(A: Mat, B: Mat): Mat {
  return matMul(transpose(A), B);
}

/** Dense inverse via Gauss-Jordan with partial pivoting. */
function inverse(A: Mat): Mat {
  const n = A.length;
  const aug: Mat = A.map((row, i) => row.concat(eye(n)[i]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
    }
    if (Math.abs(aug[pivot][col]) < 1e-14) {
      throw new Error("Matrix is singular and cannot be inverted");
    }
    const tmp = aug[col];
    aug[col] = aug[pivot];
    aug[pivot] = tmp;

    const d = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= d;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }
  return aug.map((row) => row.slice(n));
}

/** Force exact symmetry; guards against drift in the value-function recursion. */
function symmetrize(A: Mat): Mat {
  const n = A.length;
  const out = zerosMat(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) out[i][j] = 0.5 * (A[i][j] + A[j][i]);
  return out;
}
