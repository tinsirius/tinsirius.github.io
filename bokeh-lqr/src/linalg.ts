/**
 * Minimal dense linear algebra helpers, sized for the small state/input
 * dimensions used by the LQR demo (n up to a handful, scalar/vector control).
 */

type Matrix = number[][];

function zeros(rows: number, cols: number): Matrix {
  const m: Matrix = [];
  for (let i = 0; i < rows; i++) {
    m.push(new Array(cols).fill(0));
  }
  return m;
}

function identity(n: number): Matrix {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m[i][i] = 1;
  return m;
}

function dims(A: Matrix): [number, number] {
  const rows = A.length;
  const cols = rows > 0 ? A[0].length : 0;
  return [rows, cols];
}

function transpose(A: Matrix): Matrix {
  const [rows, cols] = dims(A);
  const T = zeros(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      T[j][i] = A[i][j];
    }
  }
  return T;
}

function matMul(A: Matrix, B: Matrix): Matrix {
  const [aRows, aCols] = dims(A);
  const [bRows, bCols] = dims(B);
  if (aCols !== bRows) {
    throw new Error(`Cannot multiply matrices of size ${aRows}x${aCols} and ${bRows}x${bCols}`);
  }
  const result = zeros(aRows, bCols);
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bCols; j++) {
      let sum = 0;
      for (let k = 0; k < aCols; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

function matAdd(A: Matrix, B: Matrix): Matrix {
  const [rows, cols] = dims(A);
  const [bRows, bCols] = dims(B);
  if (rows !== bRows || cols !== bCols) {
    throw new Error(`Cannot add matrices of size ${rows}x${cols} and ${bRows}x${bCols}`);
  }
  const result = zeros(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[i][j] = A[i][j] + B[i][j];
    }
  }
  return result;
}

function matSub(A: Matrix, B: Matrix): Matrix {
  const [rows, cols] = dims(A);
  const [bRows, bCols] = dims(B);
  if (rows !== bRows || cols !== bCols) {
    throw new Error(`Cannot subtract matrices of size ${rows}x${cols} and ${bRows}x${bCols}`);
  }
  const result = zeros(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[i][j] = A[i][j] - B[i][j];
    }
  }
  return result;
}

/** Gauss-Jordan inversion. Square matrices only. */
function matInverse(A: Matrix): Matrix {
  const [n, cols] = dims(A);
  if (n !== cols) {
    throw new Error(`Cannot invert a non-square matrix (${n}x${cols})`);
  }
  // Augment [A | I]
  const aug: Matrix = A.map((row, i) => [...row, ...identity(n)[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let pivotRow = col;
    let maxAbs = Math.abs(aug[col][col]);
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(aug[r][col]) > maxAbs) {
        maxAbs = Math.abs(aug[r][col]);
        pivotRow = r;
      }
    }
    if (maxAbs < 1e-12) {
      throw new Error("Matrix is singular and cannot be inverted");
    }
    if (pivotRow !== col) {
      const tmp = aug[col];
      aug[col] = aug[pivotRow];
      aug[pivotRow] = tmp;
    }
    const pivot = aug[col][col];
    for (let j = 0; j < 2 * n; j++) {
      aug[col][j] /= pivot;
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r][col];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) {
        aug[r][j] -= factor * aug[col][j];
      }
    }
  }

  return aug.map((row) => row.slice(n));
}

function scalarMul(s: number, A: Matrix): Matrix {
  return A.map((row) => row.map((v) => v * s));
}
