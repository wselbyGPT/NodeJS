// William Selby

#!/usr/bin/env node
/**
 * oscilloscope_cli.js — rolling ASCII oscilloscope with 1D (time) and 2D (X–Y) modes
 * + simulators for Mass–Spring–Damper, Lorenz, Double Pendulum, and Euler (Cornu) spiral.
 *
 * Ports the Python oscilloscope_cli.py to Node.js (no external deps).
 *
 * Keys
 * ----
 * q      quit
 * space  pause/resume
 * a      autoscale on/off
 * g      grid on/off
 * c      color on/off
 * r      reset
 *
 * Lorenz: 1/2/3 switch component/pair (1D x/y/z; XY xy/xz/yz)
 */

const { performance } = require('perf_hooks');

// ---------- Utilities ----------
function now() {
  return performance.now() / 1000; // seconds
}

function clamp(v, a, b) {
  return v < a ? a : (v > b ? b : v);
}

// ---------- Safe-ish expression evaluator ----------
class SafeExpr {
  constructor(expr) {
    this.expr = expr || '0';
    this.fn = this._compile(this.expr);
  }

  _compile(expr) {
    const allowedNames = {
      pi: Math.PI,
      e: Math.E,
      sin: Math.sin,
      cos: Math.cos,
      tan: Math.tan,
      asin: Math.asin,
      acos: Math.acos,
      atan: Math.atan,
      atan2: Math.atan2,
      sinh: Math.sinh,
      cosh: Math.cosh,
      tanh: Math.tanh,
      exp: Math.exp,
      log: Math.log,
      log10: Math.log10,
      sqrt: Math.sqrt,
      floor: Math.floor,
      ceil: Math.ceil,
      fabs: Math.abs,
      abs: Math.abs,
      pow: Math.pow
    };

    // Basic identifier whitelist: only allow t and the math names above.
    const idRegex = /[A-Za-z_]\w*/g;
    let m;
    while ((m = idRegex.exec(expr)) !== null) {
      const id = m[0];
      if (id === 't') continue;
      if (!(id in allowedNames)) {
        throw new Error(
          `Name '${id}' not allowed; only 't' and math functions/constants (sin, cos, pi, ...).`
        );
      }
    }

    const body = `with (env) { return ${expr}; }`;
    const compiled = new Function('env', body);

    return (t) => {
      const env = Object.assign({ t }, allowedNames);
      return compiled(env);
    };
  }

  eval(t) {
    return Number(this.fn(t));
  }
}

// ---------- Signal sources ----------

// 1D expression y(t)
class ExprSource1D {
  constructor(expr) {
    this.expr = expr;
    this.t = 0;
  }
  reset() {
    this.t = 0;
  }
  step(dt) {
    if (dt <= 0) dt = 1e-3;
    this.t += dt;
    return { t: this.t, value: this.expr.eval(this.t) };
  }
}

// XY expression (x(t), y(t))
class ExprSourceXY {
  constructor(xexpr, yexpr) {
    this.xexpr = xexpr;
    this.yexpr = yexpr;
    this.t = 0;
  }
  reset() {
    this.t = 0;
  }
  step(dt) {
    if (dt <= 0) dt = 1e-3;
    this.t += dt;
    return {
      t: this.t,
      value: [this.xexpr.eval(this.t), this.yexpr.eval(this.t)]
    };
  }
}

// Mass–Spring–Damper: m x'' + c x' + k x = F(t)
class MSDSource1D {
  constructor(opts) {
    this.m = Number(opts.m ?? 1.0);
    this.c = Number(opts.c ?? 0.2);
    this.k = Number(opts.k ?? 10.0);
    this.x = Number(opts.x0 ?? 0.0);
    this.v = Number(opts.v0 ?? 0.0);
    this.xi = this.x;
    this.vi = this.v;
    this.force = opts.forceExpr instanceof SafeExpr ? opts.forceExpr : new SafeExpr('0');
    this.t = 0;
  }
  reset() {
    this.x = this.xi;
    this.v = this.vi;
    this.t = 0;
  }
  _accel(t, x, v) {
    const F = this.force.eval(t);
    return (F - this.c * v - this.k * x) / this.m;
  }
  _rk4(t, h) {
    const x = this.x;
    const v = this.v;

    const a1 = this._accel(t, x, v);
    const k1x = v;
    const k1v = a1;

    const a2 = this._accel(t + 0.5 * h, x + 0.5 * h * k1x, v + 0.5 * h * k1v);
    const k2x = v + 0.5 * h * k1v;
    const k2v = a2;

    const a3 = this._accel(t + 0.5 * h, x + 0.5 * h * k2x, v + 0.5 * h * k2v);
    const k3x = v + 0.5 * h * k2v;
    const k3v = a3;

    const a4 = this._accel(t + h, x + h * k3x, v + h * k3v);
    const k4x = v + h * k3v;
    const k4v = a4;

    this.x = x + (h / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    this.v = v + (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
  }
  step(dt) {
    if (dt <= 0) dt = 1e-3;
    const maxStep = 1 / 600;
    const n = Math.max(1, Math.floor(dt / maxStep));
    const h = dt / n;
    for (let i = 0; i < n; i++) {
      this._rk4(this.t, h);
      this.t += h;
    }
    return { t: this.t, value: this.x };
  }
}

// MSD phase portrait: (x, v)
class MSDSourceXY {
  constructor(base) {
    this.base = base;
  }
  reset() {
    this.base.reset();
  }
  step(dt) {
    const { t } = this.base.step(dt);
    return { t, value: [this.base.x, this.base.v] };
  }
}

// Lorenz system
class LorenzSource1D {
  constructor(opts) {
    this.sigma = Number(opts.sigma ?? 10.0);
    this.rho = Number(opts.rho ?? 28.0);
    this.beta = Number(opts.beta ?? 8.0 / 3.0);

    this.x = Number(opts.x0_l ?? 1.0);
    this.y = Number(opts.y0_l ?? 1.0);
    this.z = Number(opts.z0_l ?? 1.0);

    this.xi = this.x;
    this.yi = this.y;
    this.zi = this.z;

    this.component = (opts.component || 'x').toLowerCase();
    this.t = 0;
  }
  reset() {
    this.x = this.xi;
    this.y = this.yi;
    this.z = this.zi;
    this.t = 0;
  }
  _deriv(x, y, z) {
    const dx = this.sigma * (y - x);
    const dy = x * (this.rho - z) - y;
    const dz = x * y - this.beta * z;
    return [dx, dy, dz];
  }
  _rk4(h) {
    const x = this.x;
    const y = this.y;
    const z = this.z;

    const [k1x, k1y, k1z] = this._deriv(x, y, z);
    const [k2x, k2y, k2z] = this._deriv(
      x + 0.5 * h * k1x,
      y + 0.5 * h * k1y,
      z + 0.5 * h * k1z
    );
    const [k3x, k3y, k3z] = this._deriv(
      x + 0.5 * h * k2x,
      y + 0.5 * h * k2y,
      z + 0.5 * h * k2z
    );
    const [k4x, k4y, k4z] = this._deriv(
      x + h * k3x,
      y + h * k3y,
      z + h * k3z
    );

    this.x = x + (h / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    this.y = y + (h / 6) * (k1y + 2 * k2y + 2 * k3y + k4y);
    this.z = z + (h / 6) * (k1z + 2 * k2z + 2 * k3z + k4z);
  }
  step(dt) {
    if (dt <= 0) dt = 1e-3;
    const maxStep = 1 / 2000;
    const n = Math.max(1, Math.floor(dt / maxStep));
    const h = dt / n;
    for (let i = 0; i < n; i++) {
      this._rk4(h);
      this.t += h;
    }
    const compMap = { x: this.x, y: this.y, z: this.z };
    const val = compMap[this.component] ?? this.x;
    return { t: this.t, value: val };
  }
}

class LorenzSourceXY {
  constructor(base, pair = 'xy') {
    this.base = base;
    this.pair = pair;
  }
  reset() {
    this.base.reset();
  }
  step(dt) {
    const { t } = this.base.step(dt);
    let x, y;
    if (this.pair === 'xz') {
      x = this.base.x;
      y = this.base.z;
    } else if (this.pair === 'yz') {
      x = this.base.y;
      y = this.base.z;
    } else {
      x = this.base.x;
      y = this.base.y;
    }
    return { t, value: [x, y] };
  }
}

// Double pendulum
class DoublePendulum1D {
  constructor(opts) {
    this.L1 = Number(opts.L1 ?? 1.0);
    this.L2 = Number(opts.L2 ?? 1.0);
    this.m1 = Number(opts.m1 ?? 1.0);
    this.m2 = Number(opts.m2 ?? 1.0);
    this.g = Number(opts.g ?? 9.81);

    this.t1 = Number(opts.t1 ?? Math.PI / 2);
    this.t2 = Number(opts.t2 ?? Math.PI / 2);
    this.w1 = Number(opts.w1 ?? 0.0);
    this.w2 = Number(opts.w2 ?? 0.0);

    this.t1i = this.t1;
    this.t2i = this.t2;
    this.w1i = this.w1;
    this.w2i = this.w2;

    const comp = (opts.component || 't1').toLowerCase();
    this.component = (comp === 't2') ? 't2' : 't1';
    this.t = 0;
  }

  reset() {
    this.t1 = this.t1i;
    this.t2 = this.t2i;
    this.w1 = this.w1i;
    this.w2 = this.w2i;
    this.t = 0;
  }

  _accels(t1, t2, w1, w2) {
    const g = this.g;
    const m1 = this.m1;
    const m2 = this.m2;
    const L1 = this.L1;
    const L2 = this.L2;
    const d = t1 - t2;

    const den = 2 * m1 + m2 - m2 * Math.cos(2 * d);
    const a1 = (
      -g * (2 * m1 + m2) * Math.sin(t1) -
      m2 * g * Math.sin(t1 - 2 * t2) -
      2 * Math.sin(d) * m2 * (w2 * w2 * L2 + w1 * w1 * L1 * Math.cos(d))
    ) / (L1 * den);

    const a2 = (
      2 * Math.sin(d) *
      (w1 * w1 * L1 * (m1 + m2) +
        g * (m1 + m2) * Math.cos(t1) +
        w2 * w2 * L2 * m2 * Math.cos(d))
    ) / (L2 * den);

    return [a1, a2];
  }

  _rk4(h) {
    const f = (y) => {
      const [t1, t2, w1, w2] = y;
      const [a1, a2] = this._accels(t1, t2, w1, w2);
      return [w1, w2, a1, a2];
    };

    const y0 = [this.t1, this.t2, this.w1, this.w2];

    const k1 = f(y0);
    const y1 = y0.map((v, i) => v + 0.5 * h * k1[i]);

    const k2 = f(y1);
    const y2 = y0.map((v, i) => v + 0.5 * h * k2[i]);

    const k3 = f(y2);
    const y3 = y0.map((v, i) => v + h * k3[i]);

    const k4 = f(y3);

    this.t1 = y0[0] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    this.t2 = y0[1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    this.w1 = y0[2] + (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
    this.w2 = y0[3] + (h / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]);
  }

  step(dt) {
    if (dt <= 0) dt = 1e-3;
    const maxStep = 1 / 2000;
    const n = Math.max(1, Math.floor(dt / maxStep));
    const h = dt / n;
    for (let i = 0; i < n; i++) {
      this._rk4(h);
      this.t += h;
    }
    const val = this.component === 't2' ? this.t2 : this.t1;
    return { t: this.t, value: val };
  }

  pos() {
    const x1 = this.L1 * Math.sin(this.t1);
    const y1 = -this.L1 * Math.cos(this.t1);
    const x2 = x1 + this.L2 * Math.sin(this.t2);
    const y2 = y1 - this.L2 * Math.cos(this.t2);
    return [x1, y1, x2, y2];
  }
}

class DoublePendulumXY {
  constructor(base, which = 'tip') {
    this.base = base;
    this.which = which;
  }
  reset() {
    this.base.reset();
  }
  step(dt) {
    const { t } = this.base.step(dt);
    const [x1, y1, x2, y2] = this.base.pos();
    if (this.which === 'tip') {
      return { t, value: [x2, y2] };
    }
    return { t, value: [x1, y1] };
  }
}

// Euler (Cornu) spiral
class EulerSpiralXY {
  constructor(speed = 0.8) {
    this.speed = Number(speed);
    this.s = 0;
    this.x = 0;
    this.y = 0;
  }
  reset() {
    this.s = 0;
    this.x = 0;
    this.y = 0;
  }
  static _f(s) {
    const a = 0.5 * Math.PI * s * s;
    return [Math.cos(a), Math.sin(a)];
  }
  _rk4Step(h) {
    const [k1x, k1y] = EulerSpiralXY._f(this.s);
    const [k2x, k2y] = EulerSpiralXY._f(this.s + 0.5 * h);
    const [k3x, k3y] = EulerSpiralXY._f(this.s + 0.5 * h);
    const [k4x, k4y] = EulerSpiralXY._f(this.s + h);

    this.x += (h / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    this.y += (h / 6) * (k1y + 2 * k2y + 2 * k3y + k4y);
    this.s += h;
  }
  step(dt) {
    if (dt <= 0) dt = 1e-3;
    const ds = this.speed * dt;
    const maxH = 0.01;
    const n = Math.max(1, Math.floor(Math.abs(ds) / maxH));
    const h = ds / n;
    for (let i = 0; i < n; i++) {
      this._rk4Step(h);
    }
    return { t: this.s, value: [this.x, this.y] };
  }
}

// ---------- Color ramp (ANSI 256) ----------
class ColorRamp {
  constructor() {
    this.enabled = process.stdout.isTTY;
    this.codes = [];
    if (this.enabled) {
      // Similar-ish blue→cyan→green ramp
      const ramp = [21, 27, 33, 39, 45, 51, 50, 49, 48, 47, 46];
      this.codes = ramp.map((fg) => `\x1b[38;5;${fg}m`);
    }
  }
  attrForFraction(frac) {
    if (!this.enabled || this.codes.length < 2) return '';
    let f = frac;
    if (f < 0) f = 0;
    if (f > 1) f = 1;
    const idx = Math.max(
      0,
      Math.min(this.codes.length - 1, Math.round(f * (this.codes.length - 1)))
    );
    return this.codes[idx];
  }
}

// ---------- Scopes ----------
class Scope1D {
  constructor(source, opts) {
    this.source = source;
    this.fps = Math.max(1, parseInt(opts.fps ?? 60, 10));
    this.autoscale = !opts.noAutoscale;
    this.grid = !opts.noGrid;
    this.colorOn = !opts.noColor;
    this.ramp = new ColorRamp();
    this.values = [];
    this.paused = false;
    this._t = 0;
  }

  _scale() {
    if (!this.autoscale || this.values.length === 0) {
      return { ymin: -1, ymax: 1 };
    }
    let vmin = Infinity;
    let vmax = -Infinity;
    for (const v of this.values) {
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
    if (vmin === Infinity) return { ymin: -1, ymax: 1 };
    if (vmin === vmax) {
      const pad = vmin === 0 ? 1 : Math.abs(vmin) * 0.2;
      return { ymin: vmin - pad, ymax: vmax + pad };
    }
    const span = vmax - vmin;
    const pad = 0.1 * span;
    return { ymin: vmin - pad, ymax: vmax + pad };
  }

  _draw(height, width) {
    const h = height;
    const w = width;
    const grid = [];
    for (let r = 0; r < h; r++) {
      const row = new Array(w).fill(' ');
      grid.push(row);
    }

    const { ymin, ymax } = this._scale();
    const span = ymax - ymin || 1e-9;

    const yToRow = (y) => {
      const frac = (y - ymin) / span;
      const row = Math.round((1 - frac) * (h - 2));
      return clamp(row, 0, Math.max(0, h - 2));
    };

    // Grid
    if (this.grid && h >= 4 && w >= 20) {
      if (ymin < 0 && 0 < ymax) {
        const z = yToRow(0);
        for (let x = 0; x < w; x++) {
          grid[z][x] = '-';
        }
      }
      for (let x = 0; x < w; x += 10) {
        for (let r = 0; r < h - 1; r += 2) {
          grid[r][x] = '|';
        }
      }
    }

    // Trace
    if (this.values.length > 0) {
      let vmin = Infinity;
      let vmax = -Infinity;
      for (const v of this.values) {
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
      if (vmin === Infinity) {
        vmin = 0;
        vmax = 1;
      }
      const denom = vmax - vmin || 1;
      const n = this.values.length;
      for (let x = 0; x < n && x < w; x++) {
        const y = this.values[n - w + x] ?? this.values[x];
        const row = yToRow(y);
        let ch = '*';
        if (this.colorOn && this.ramp.enabled) {
          const frac = (y - vmin) / denom;
          const color = this.ramp.attrForFraction(frac);
          ch = color + '*' + '\x1b[0m';
        }
        grid[row][x] = ch;
      }
    }

    // Status line
    const status =
      `1D  t=${this._t.toFixed(3).padStart(7, ' ')}s  ` +
      `autoscale=${this.autoscale ? 'ON' : 'OFF'}  ` +
      `grid=${this.grid ? 'ON' : 'OFF'}  ` +
      `fps=${this.fps}`;
    const statusRow = h - 1;
    const statusText = status.slice(0, w);
    for (let i = 0; i < statusText.length; i++) {
      grid[statusRow][i] = statusText[i];
    }

    const out = grid.map((row) => row.join('')).join('\n');
    process.stdout.write('\x1b[2J\x1b[H' + out);
  }

  tick(dt) {
    const width = Math.max(10, process.stdout.columns || 80);
    const height = Math.max(5, process.stdout.rows || 24);

    if (!this.paused) {
      const { t, value } = this.source.step(dt);
      this._t = t;
      let y = value;
      if (!Number.isFinite(y)) y = 0;
      this.values.push(y);
      while (this.values.length > width) {
        this.values.shift();
      }
    }

    this._draw(height, width);
  }
}

class ScopeXY {
  constructor(source, opts) {
    this.source = source;
    this.fps = Math.max(1, parseInt(opts.fps ?? 60, 10));
    this.autoscale = !opts.noAutoscale;
    this.grid = !opts.noGrid;
    this.colorOn = !opts.noColor;
    this.trailFactor = Number(opts.trailFactor ?? 2.0);
    this.trail = [];
    this.ramp = new ColorRamp();
    this.paused = false;
    this._t = 0;
    this.fadeChars = '·.:oO@'; // oldest→newest
  }

  _resize(width) {
    const maxLen = Math.max(20, Math.floor(width * this.trailFactor));
    while (this.trail.length > maxLen) {
      this.trail.shift();
    }
  }

  _scale() {
    if (!this.autoscale || this.trail.length === 0) {
      return { xmin: -1, xmax: 1, ymin: -1, ymax: 1 };
    }
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const [x, y] of this.trail) {
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    if (xmin === xmax) {
      const pad = xmin === 0 ? 1 : Math.abs(xmin) * 0.2;
      xmin -= pad;
      xmax += pad;
    }
    if (ymin === ymax) {
      const pad = ymin === 0 ? 1 : Math.abs(ymin) * 0.2;
      ymin -= pad;
      ymax += pad;
    }
    const dx = xmax - xmin;
    const dy = ymax - ymin;
    return {
      xmin: xmin - 0.1 * dx,
      xmax: xmax + 0.1 * dx,
      ymin: ymin - 0.1 * dy,
      ymax: ymax + 0.1 * dy
    };
  }

  _draw(height, width) {
    const h = height;
    const w = width;
    const grid = [];
    for (let r = 0; r < h; r++) {
      const row = new Array(w).fill(' ');
      grid.push(row);
    }

    const { xmin, xmax, ymin, ymax } = this._scale();
    const xSpan = xmax - xmin || 1e-12;
    const ySpan = ymax - ymin || 1e-12;

    const xToCol = (x) => {
      const frac = (x - xmin) / xSpan;
      const col = Math.round(frac * (w - 1));
      return clamp(col, 0, Math.max(0, w - 1));
    };
    const yToRow = (y) => {
      const frac = (y - ymin) / ySpan;
      const row = Math.round((1 - frac) * (h - 2));
      return clamp(row, 0, Math.max(0, h - 2));
    };

    const plot = (c, r, ch) => {
      if (r < 0 || r >= h - 1 || c < 0 || c >= w) return;
      grid[r][c] = ch;
    };

    const line = (c0, r0, c1, r1, ch) => {
      let x0 = c0;
      let y0 = r0;
      const x1 = c1;
      const y1 = r1;
      const dx = Math.abs(x1 - x0);
      const sx = x0 < x1 ? 1 : -1;
      const dy = -Math.abs(y1 - y0);
      const sy = y0 < y1 ? 1 : -1;
      let err = dx + dy;
      while (true) {
        plot(x0, y0, ch);
        if (x0 === x1 && y0 === y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) {
          err += dy;
          x0 += sx;
        }
        if (e2 <= dx) {
          err += dx;
          y0 += sy;
        }
      }
    };

    // Grid axes
    if (this.grid && h >= 4 && w >= 20) {
      if (xmin < 0 && 0 < xmax) {
        const x0 = xToCol(0);
        for (let r = 0; r < h - 1; r++) {
          grid[r][x0] = '|';
        }
      }
      if (ymin < 0 && 0 < ymax) {
        const y0 = yToRow(0);
        for (let c = 0; c < w; c++) {
          grid[y0][c] = '-';
        }
      }
    }

    // Trail
    const n = this.trail.length;
    if (n === 1) {
      const [x, y] = this.trail[0];
      const c = xToCol(x);
      const r = yToRow(y);
      plot(c, r, this.fadeChars[this.fadeChars.length - 1]);
    } else if (n > 1) {
      for (let i = 1; i < n; i++) {
        const [x0, y0] = this.trail[i - 1];
        const [x1, y1] = this.trail[i];
        const c0 = xToCol(x0);
        const r0 = yToRow(y0);
        const c1 = xToCol(x1);
        const r1 = yToRow(y1);
        const age = i / (n - 1);
        const idx = Math.min(
          this.fadeChars.length - 1,
          Math.floor(age * (this.fadeChars.length - 1))
        );
        let ch = this.fadeChars[idx];
        if (this.colorOn && this.ramp.enabled) {
          const color = this.ramp.attrForFraction(age);
          ch = color + ch + '\x1b[0m';
        }
        line(c0, r0, c1, r1, ch);
      }
    }

    // Status
    const status =
      `XY  t=${this._t.toFixed(3).padStart(7, ' ')}s  ` +
      `trail=${n}  ` +
      `autoscale=${this.autoscale ? 'ON' : 'OFF'}  ` +
      `grid=${this.grid ? 'ON' : 'OFF'}  ` +
      `fps=${this.fps}`;
    const statusRow = h - 1;
    const statusText = status.slice(0, w);
    for (let i = 0; i < statusText.length; i++) {
      grid[statusRow][i] = statusText[i];
    }

    const out = grid.map((row) => row.join('')).join('\n');
    process.stdout.write('\x1b[2J\x1b[H' + out);
  }

  tick(dt) {
    const width = Math.max(10, process.stdout.columns || 80);
    const height = Math.max(5, process.stdout.rows || 24);
    this._resize(width);

    if (!this.paused) {
      const { t, value } = this.source.step(dt);
      this._t = t;
      const [x, y] = value;
      if (Number.isFinite(x) && Number.isFinite(y)) {
        this.trail.push([x, y]);
      }
    }

    this._draw(height, width);
  }
}

// ---------- CLI parsing ----------
function printUsageAndExit(msg) {
  if (msg) {
    console.error('Error:', msg);
  }
  console.error(`
Usage:
  node oscilloscope_cli.js --expr "sin(2*pi*2*t)" [--fps 60]
  node oscilloscope_cli.js --xy --xexpr "sin(2*pi*3*t)" --yexpr "sin(2*pi*2*t)" --fps 60
  node oscilloscope_cli.js --msd [--xy --msd-phase] [params...]
  node oscilloscope_cli.js --lorenz [--xy --xy-components xy|xz|yz]
  node oscilloscope_cli.js --pendulum [--xy]
  node oscilloscope_cli.js --euler --xy

Keys: q quit · space pause/resume · a autoscale · g grid · c color · r reset
Lorenz: 1/2/3 switch component/pair (1D x/y/z; XY xy/xz/yz)
`);
  process.exit(1);
}

function parseArgs(argv) {
  const args = {
    expr: null,
    msd: false,
    lorenz: false,
    pendulum: false,
    euler: false,
    xy: false,
    xexpr: null,
    yexpr: null,
    msdPhase: false,
    xyComponents: 'xy',
    m: 1.0,
    c: 0.2,
    k: 10.0,
    x0: 0.0,
    v0: 0.0,
    forceExpr: '0',
    sigma: 10.0,
    rho: 28.0,
    beta: 8.0 / 3.0,
    x0_l: 1.0,
    y0_l: 1.0,
    z0_l: 1.0,
    component: 'x',
    L1: 1.0,
    L2: 1.0,
    m1: 1.0,
    m2: 1.0,
    g: 9.81,
    t1: Math.PI / 2,
    t2: Math.PI / 2,
    w1: 0.0,
    w2: 0.0,
    eulerSpeed: 0.8,
    fps: 60,
    noAutoscale: false,
    noGrid: false,
    noColor: false,
    trailFactor: 2.0
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      printUsageAndExit();
    } else if (arg === '--expr') {
      if (i + 1 >= argv.length) printUsageAndExit('--expr requires an argument');
      args.expr = argv[++i];
    } else if (arg === '--msd') {
      args.msd = true;
    } else if (arg === '--lorenz') {
      args.lorenz = true;
    } else if (arg === '--pendulum') {
      args.pendulum = true;
    } else if (arg === '--euler') {
      args.euler = true;
    } else if (arg === '--xy') {
      args.xy = true;
    } else if (arg === '--xexpr') {
      if (i + 1 >= argv.length) printUsageAndExit('--xexpr requires an argument');
      args.xexpr = argv[++i];
    } else if (arg === '--yexpr') {
      if (i + 1 >= argv.length) printUsageAndExit('--yexpr requires an argument');
      args.yexpr = argv[++i];
    } else if (arg === '--msd-phase') {
      args.msdPhase = true;
    } else if (arg === '--xy-components') {
      if (i + 1 >= argv.length) printUsageAndExit('--xy-components requires an argument');
      args.xyComponents = argv[++i];
    } else if (arg === '--m') {
      args.m = parseFloat(argv[++i]);
    } else if (arg === '--c') {
      args.c = parseFloat(argv[++i]);
    } else if (arg === '--k') {
      args.k = parseFloat(argv[++i]);
    } else if (arg === '--x0') {
      args.x0 = parseFloat(argv[++i]);
    } else if (arg === '--v0') {
      args.v0 = parseFloat(argv[++i]);
    } else if (arg === '--force-expr') {
      args.forceExpr = argv[++i];
    } else if (arg === '--sigma') {
      args.sigma = parseFloat(argv[++i]);
    } else if (arg === '--rho') {
      args.rho = parseFloat(argv[++i]);
    } else if (arg === '--beta') {
      args.beta = parseFloat(argv[++i]);
    } else if (arg === '--x0_l') {
      args.x0_l = parseFloat(argv[++i]);
    } else if (arg === '--y0_l') {
      args.y0_l = parseFloat(argv[++i]);
    } else if (arg === '--z0_l') {
      args.z0_l = parseFloat(argv[++i]);
    } else if (arg === '--component') {
      args.component = argv[++i];
    } else if (arg === '--L1') {
      args.L1 = parseFloat(argv[++i]);
    } else if (arg === '--L2') {
      args.L2 = parseFloat(argv[++i]);
    } else if (arg === '--m1') {
      args.m1 = parseFloat(argv[++i]);
    } else if (arg === '--m2') {
      args.m2 = parseFloat(argv[++i]);
    } else if (arg === '--g') {
      args.g = parseFloat(argv[++i]);
    } else if (arg === '--t1') {
      args.t1 = parseFloat(argv[++i]);
    } else if (arg === '--t2') {
      args.t2 = parseFloat(argv[++i]);
    } else if (arg === '--w1') {
      args.w1 = parseFloat(argv[++i]);
    } else if (arg === '--w2') {
      args.w2 = parseFloat(argv[++i]);
    } else if (arg === '--euler-speed') {
      args.eulerSpeed = parseFloat(argv[++i]);
    } else if (arg === '--fps') {
      args.fps = parseInt(argv[++i], 10);
    } else if (arg === '--no-autoscale') {
      args.noAutoscale = true;
    } else if (arg === '--no-grid') {
      args.noGrid = true;
    } else if (arg === '--no-color') {
      args.noColor = true;
    } else if (arg === '--trail-factor') {
      args.trailFactor = parseFloat(argv[++i]);
    } else {
      printUsageAndExit(`Unknown argument: ${arg}`);
    }
  }

  return args;
}

// ---------- Build source & scope ----------
function buildSourceAndScope(args) {
  let source = null;
  let scopeKind = null;

  const anySim = args.msd || args.lorenz || args.pendulum || args.euler;

  if (args.xy && !anySim && args.xexpr && args.yexpr) {
    // Pure XY expressions
    source = new ExprSourceXY(new SafeExpr(args.xexpr), new SafeExpr(args.yexpr));
    scopeKind = 'xy';
  } else if (args.expr && !args.xy) {
    // 1D expression
    source = new ExprSource1D(new SafeExpr(args.expr));
    scopeKind = '1d';
  } else if (args.msd && !args.xy) {
    const msd = new MSDSource1D({
      m: args.m,
      c: args.c,
      k: args.k,
      x0: args.x0,
      v0: args.v0,
      forceExpr: new SafeExpr(args.forceExpr)
    });
    source = msd;
    scopeKind = '1d';
  } else if (args.msd && args.xy) {
    if (!args.msdPhase) {
      printUsageAndExit('For --msd + --xy you must pass --msd-phase to plot (x,v)');
    }
    const msd = new MSDSource1D({
      m: args.m,
      c: args.c,
      k: args.k,
      x0: args.x0,
      v0: args.v0,
      forceExpr: new SafeExpr(args.forceExpr)
    });
    source = new MSDSourceXY(msd);
    scopeKind = 'xy';
  } else if (args.lorenz && !args.xy) {
    source = new LorenzSource1D({
      sigma: args.sigma,
      rho: args.rho,
      beta: args.beta,
      x0_l: args.x0_l,
      y0_l: args.y0_l,
      z0_l: args.z0_l,
      component: args.component
    });
    scopeKind = '1d';
  } else if (args.lorenz && args.xy) {
    const base = new LorenzSource1D({
      sigma: args.sigma,
      rho: args.rho,
      beta: args.beta,
      x0_l: args.x0_l,
      y0_l: args.y0_l,
      z0_l: args.z0_l,
      component: 'x'
    });
    source = new LorenzSourceXY(base, args.xyComponents);
    scopeKind = 'xy';
  } else if (args.pendulum && !args.xy) {
    source = new DoublePendulum1D({
      L1: args.L1,
      L2: args.L2,
      m1: args.m1,
      m2: args.m2,
      g: args.g,
      t1: args.t1,
      t2: args.t2,
      w1: args.w1,
      w2: args.w2,
      component: (args.component === 't2' ? 't2' : 't1')
    });
    scopeKind = '1d';
  } else if (args.pendulum && args.xy) {
    const base = new DoublePendulum1D({
      L1: args.L1,
      L2: args.L2,
      m1: args.m1,
      m2: args.m2,
      g: args.g,
      t1: args.t1,
      t2: args.t2,
      w1: args.w1,
      w2: args.w2,
      component: 't1'
    });
    source = new DoublePendulumXY(base, 'tip');
    scopeKind = 'xy';
  } else if (args.euler && args.xy) {
    source = new EulerSpiralXY(args.eulerSpeed);
    scopeKind = 'xy';
  } else {
    printUsageAndExit(
      'Choose a mode: --expr (1D), or --xy with --xexpr/--yexpr, or --msd, or --lorenz, or --pendulum, or --euler'
    );
  }

  return { source, scopeKind };
}

// ---------- Main ----------
function main() {
  const argv = process.argv.slice(2);
  const args = parseArgs(argv);
  const { source, scopeKind } = buildSourceAndScope(args);

  const scope =
    scopeKind === '1d'
      ? new Scope1D(source, args)
      : new ScopeXY(source, args);

  // Terminal setup
  process.stdout.write('\x1b[?25l'); // hide cursor
  let exiting = false;

  const cleanup = () => {
    if (exiting) return;
    exiting = true;
    clearInterval(timer);
    process.stdout.write('\x1b[0m\x1b[2J\x1b[H'); // reset colors & clear
    try {
      if (process.stdin.isTTY) {
        process.stdin.setRawMode(false);
      }
    } catch (_e) {}
    process.stdin.pause();
    process.stdout.write('\x1b[?25h'); // show cursor
  };

  process.on('SIGINT', () => {
    cleanup();
    process.exit(0);
  });

  // Keyboard input
  if (process.stdin.isTTY) {
    process.stdin.setRawMode(true);
  }
  process.stdin.setEncoding('utf8');
  process.stdin.resume();

  process.stdin.on('data', (data) => {
    for (const ch of data) {
      if (ch === '\u0003') {
        // Ctrl-C
        cleanup();
        process.exit(0);
      } else if (ch === 'q' || ch === 'Q') {
        cleanup();
        process.exit(0);
      } else if (ch === ' ') {
        scope.paused = !scope.paused;
      } else if (ch === 'a' || ch === 'A') {
        scope.autoscale = !scope.autoscale;
      } else if (ch === 'g' || ch === 'G') {
        scope.grid = !scope.grid;
      } else if (ch === 'c' || ch === 'C') {
        scope.colorOn = !scope.colorOn;
      } else if (ch === 'r' || ch === 'R') {
        if (typeof source.reset === 'function') {
          source.reset();
        }
        if (scopeKind === '1d') {
          scope.values = [];
        } else {
          scope.trail = [];
        }
        scope._t = 0;
      } else if (ch === '1' || ch === '2' || ch === '3') {
        // Lorenz component/pair switching
        if (scopeKind === '1d' && source instanceof LorenzSource1D) {
          const compMap = { '1': 'x', '2': 'y', '3': 'z' };
          source.component = compMap[ch] || source.component;
        } else if (
          scopeKind === 'xy' &&
          source instanceof LorenzSourceXY &&
          typeof source.pair === 'string'
        ) {
          const pairMap = { '1': 'xy', '2': 'xz', '3': 'yz' };
          source.pair = pairMap[ch] || source.pair;
        }
      }
    }
  });

  const frameInterval = 1000 / (scope.fps || 60);
  let lastTime = now();

  const timer = setInterval(() => {
    const tNow = now();
    const dt = tNow - lastTime;
    lastTime = tNow;
    scope.tick(dt);
  }, frameInterval);
}

if (require.main === module) {
  main();
}
