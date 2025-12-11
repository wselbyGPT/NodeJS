// William Selby

#!/usr/bin/env node
'use strict';

// ======================= ANSI / TTY helpers ===========================
const ansi = {
  reset: '\x1b[0m',
  clear: '\x1b[2J',
  home: '\x1b[H',
  hideCursor: '\x1b[?25l',
  showCursor: '\x1b[?25h',
  color: (code) => `\x1b[${code}m`
};

// ======================= CLI argument parsing =========================
function printHelp() {
  console.log(`
Spiral Neural Net ASCII Visualizer (Node.js)

Usage:
  node neuroscope_spiral.js [options]

Options:
  --hidden H1,H2,...   Comma-separated hidden layer sizes (default "32,32")
  --epochs N           Training epochs (default 2500)
  --lr LR              Learning rate (default 0.03)
  --points N           Points per class (default 300)
  --noise S            Noise std-dev (default 0.2)
  --width W            ASCII plot width (default 80)
  --height H           ASCII plot height (default 40)
  --refresh N          Redraw every N epochs (default 10)
  --seed S             RNG seed (default 1337)
  --no-color           Disable ANSI color output
  -h, --help           Show this help and exit

Examples:
  node neuroscope_spiral.js --hidden 16,16 --epochs 1500
  node neuroscope_spiral.js --hidden 64,64,64 --points 600 --lr 0.02 --epochs 5000 --width 100 --height 50
`);
}

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    pointsPerClass: 300,
    noise: 0.2,
    hidden: [32, 32],
    lr: 0.03,
    epochs: 2500,
    width: 80,
    height: 40,
    refreshEvery: 10,
    seed: 1337,
    noColor: false
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '-h':
      case '--help':
        printHelp();
        process.exit(0);
        break;
      case '--points':
        config.pointsPerClass = parseInt(args[++i], 10);
        break;
      case '--noise':
        config.noise = parseFloat(args[++i]);
        break;
      case '--hidden': {
        const val = args[++i] || '';
        const parts = val.split(',').map(s => parseInt(s.trim(), 10));
        const valid = parts.filter(n => Number.isFinite(n) && n > 0);
        if (valid.length > 0) config.hidden = valid;
        break;
      }
      case '--lr':
        config.lr = parseFloat(args[++i]);
        break;
      case '--epochs':
        config.epochs = parseInt(args[++i], 10);
        break;
      case '--width':
        config.width = parseInt(args[++i], 10);
        break;
      case '--height':
        config.height = parseInt(args[++i], 10);
        break;
      case '--refresh':
      case '--fps':
        config.refreshEvery = parseInt(args[++i], 10);
        break;
      case '--seed':
        config.seed = parseInt(args[++i], 10);
        break;
      case '--no-color':
      case '--no-ansi':
        config.noColor = true;
        break;
      default:
        console.error('Unknown argument:', arg);
        printHelp();
        process.exit(1);
    }
  }

  if (!Number.isFinite(config.pointsPerClass) || config.pointsPerClass <= 0) {
    config.pointsPerClass = 300;
  }
  if (!Number.isFinite(config.epochs) || config.epochs <= 0) {
    config.epochs = 2500;
  }
  if (!Number.isFinite(config.lr) || config.lr <= 0) {
    config.lr = 0.03;
  }
  if (!Number.isFinite(config.width) || config.width <= 10) {
    config.width = 80;
  }
  if (!Number.isFinite(config.height) || config.height <= 10) {
    config.height = 40;
  }
  if (!Number.isFinite(config.refreshEvery) || config.refreshEvery <= 0) {
    config.refreshEvery = 10;
  }
  if (!Number.isFinite(config.seed) || config.seed <= 0) {
    config.seed = 1337;
  }

  return config;
}

// ======================= RNG + helpers ================================
function makeRNG(seed) {
  let s = (seed >>> 0) || 1;
  return function rng() {
    // Simple LCG
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function randnFactory(rng) {
  let spare = null;
  return function randn() {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0, v = 0, r = 0;
    do {
      u = rng() * 2 - 1;
      v = rng() * 2 - 1;
      r = u * u + v * v;
    } while (!r || r >= 1);
    const c = Math.sqrt(-2 * Math.log(r) / r);
    spare = v * c;
    return u * c;
  };
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  let sum = 0;
  const exps = logits.map(v => {
    const e = Math.exp(v - maxLogit);
    sum += e;
    return e;
  });
  return exps.map(e => e / (sum || 1));
}

function shuffleInPlace(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

// ======================= Data generation =============================
// Canonical two-spiral dataset: two intertwined arms in polar coordinates.
function generateSpiral(pointsPerClass, noise, rng) {
  const randn = randnFactory(rng);
  const data = [];
  const labels = [];

  const maxRadius = 5.0;       // overall size of spiral
  const turns = 2.0;           // number of turns (2 full rotations)
  const twoPi = 2 * Math.PI;

  for (let classIx = 0; classIx < 2; classIx++) {
    for (let i = 0; i < pointsPerClass; i++) {
      const frac = i / (pointsPerClass - 1);   // 0..1
      const r = frac * maxRadius;
      let theta = turns * frac * twoPi;

      // offset second spiral by pi (180 degrees)
      if (classIx === 1) {
        theta += Math.PI;
      }

      const x = r * Math.cos(theta) + noise * randn();
      const y = r * Math.sin(theta) + noise * randn();

      data.push([x, y]);
      labels.push(classIx);
    }
  }
  return { data, labels };
}

function computeBounds(data) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of data) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!Number.isFinite(minX)) {
    minX = -1; maxX = 1; minY = -1; maxY = 1;
  }
  const padX = (maxX - minX || 1) * 0.1;
  const padY = (maxY - minY || 1) * 0.1;
  return {
    minX: minX - padX,
    maxX: maxX + padX,
    minY: minY - padY,
    maxY: maxY + padY
  };
}

// ======================= Neural network ==============================
class Network {
  constructor(inputSize, hiddenSizes, outputSize, rng) {
    this.sizes = [inputSize, ...hiddenSizes, outputSize];
    this.numLayers = this.sizes.length - 1; // number of weight layers
    this.w = [];
    this.b = [];

    for (let l = 0; l < this.numLayers; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const layerW = new Array(nOut);
      const layerB = new Array(nOut);
      const scale = Math.sqrt(2 / (nIn + nOut));

      for (let j = 0; j < nOut; j++) {
        const row = new Array(nIn);
        for (let i = 0; i < nIn; i++) {
          row[i] = (rng() * 2 - 1) * scale;
        }
        layerW[j] = row;
        layerB[j] = 0;
      }

      this.w.push(layerW);
      this.b.push(layerB);
    }
  }

  forwardDetailed(x) {
    const activations = [];
    activations[0] = x.slice();
    let a = activations[0];

    for (let l = 0; l < this.numLayers; l++) {
      const w = this.w[l];
      const b = this.b[l];
      const nOut = w.length;
      const z = new Array(nOut);

      for (let j = 0; j < nOut; j++) {
        const wj = w[j];
        let sum = b[j];
        for (let i = 0; i < wj.length; i++) {
          sum += wj[i] * a[i];
        }
        z[j] = sum;
      }

      let aNext;
      if (l < this.numLayers - 1) {
        // Hidden: tanh
        aNext = new Array(nOut);
        for (let j = 0; j < nOut; j++) {
          aNext[j] = Math.tanh(z[j]);
        }
      } else {
        // Output: logits
        aNext = z.slice();
      }

      activations[l + 1] = aNext;
      a = aNext;
    }

    const logits = activations[this.numLayers];
    const probs = softmax(logits);
    return { activations, logits, probs };
  }

  trainSample(x, y, lr) {
    const { activations, probs } = this.forwardDetailed(x);
    const L = this.numLayers;

    // Output layer gradient: dL/dz_L = softmax - one_hot
    const dZ = new Array(L + 1);
    const dZ_L = new Array(this.sizes[L]);
    for (let j = 0; j < dZ_L.length; j++) {
      dZ_L[j] = probs[j] - (j === y ? 1 : 0);
    }
    dZ[L] = dZ_L;

    // Backpropagate
    for (let l = L; l >= 1; l--) {
      const wIndex = l - 1;
      const w = this.w[wIndex];
      const b = this.b[wIndex];
      const nOut = w.length;
      const nIn = w[0].length;

      const dZ_curr = dZ[l];
      const dA_prev = new Array(nIn).fill(0);

      // dA_prev = W^T * dZ_curr
      for (let j = 0; j < nOut; j++) {
        const wj = w[j];
        const dzj = dZ_curr[j];
        for (let i = 0; i < nIn; i++) {
          dA_prev[i] += wj[i] * dzj;
        }
      }

      // Gradient step for W and b
      const aPrev = activations[l - 1]; // size nIn
      for (let j = 0; j < nOut; j++) {
        const wj = w[j];
        const dzj = dZ_curr[j];
        b[j] -= lr * dzj;
        for (let i = 0; i < nIn; i++) {
          wj[i] -= lr * dzj * aPrev[i];
        }
      }

      // Hidden: backprop through tanh
      if (l > 1) {
        const aPrevLayer = activations[l - 1];
        const dZ_prev = new Array(nIn);
        for (let i = 0; i < nIn; i++) {
          const a_i = aPrevLayer[i];
          dZ_prev[i] = dA_prev[i] * (1 - a_i * a_i);
        }
        dZ[l - 1] = dZ_prev;
      }
    }

    const loss = -Math.log(Math.max(probs[y], 1e-12));
    const pred = probs[0] > probs[1] ? 0 : 1;
    const correct = pred === y ? 1 : 0;
    return { loss, correct };
  }

  predictClass(x) {
    let a = x.slice();
    for (let l = 0; l < this.numLayers; l++) {
      const w = this.w[l];
      const b = this.b[l];
      const nOut = w.length;
      const z = new Array(nOut);

      for (let j = 0; j < nOut; j++) {
        const wj = w[j];
        let sum = b[j];
        for (let i = 0; i < wj.length; i++) {
          sum += wj[i] * a[i];
        }
        z[j] = sum;
      }

      if (l < this.numLayers - 1) {
        for (let j = 0; j < nOut; j++) {
          z[j] = Math.tanh(z[j]);
        }
      }
      a = z;
    }

    const probs = softmax(a);
    return probs[0] > probs[1] ? 0 : 1;
  }
}

// ======================= ASCII drawing ===============================
function drawFrame(config, epoch, loss, acc, net, data, labels, bounds) {
  const { width, height, noColor } = config;
  const { minX, maxX, minY, maxY } = bounds;

  const gridChar = Array.from({ length: height }, () => new Array(width).fill(' '));
  const gridColor = Array.from({ length: height }, () => new Array(width).fill(null));

  const dx = (maxX - minX) / width;
  const dy = (maxY - minY) / height;

  // Background: NN decision boundary
  for (let iy = 0; iy < height; iy++) {
    const yCoord = maxY - (iy + 0.5) * dy; // flip vertical
    for (let ix = 0; ix < width; ix++) {
      const xCoord = minX + (ix + 0.5) * dx;
      const cls = net.predictClass([xCoord, yCoord]);
      const ch = cls === 0 ? '.' : '+';
      const color = cls === 0 ? 34 : 31; // blue / red
      gridChar[iy][ix] = ch;
      gridColor[iy][ix] = color;
    }
  }

  // Overlay training points so they stand out
  const n = data.length;
  for (let idx = 0; idx < n; idx++) {
    const [x, y] = data[idx];
    const label = labels[idx];

    let ix = Math.floor((x - minX) / (maxX - minX) * width);
    let iy = Math.floor((maxY - y) / (maxY - minY) * height);

    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
      const ch = label === 0 ? 'o' : 'x';
      const color = label === 0 ? 36 : 33; // cyan / yellow
      gridChar[iy][ix] = ch;
      gridColor[iy][ix] = color;
    }
  }

  let output = '';
  output += ansi.clear + ansi.home;
  output += `Spiral Neural Net ASCII Visualizer (Node.js)\n`;
  output += `Epoch ${epoch} / ${config.epochs}  |  Loss: ${loss.toFixed(4)}  |  Acc: ${(acc * 100).toFixed(1)}%  | Hidden: [${config.hidden.join(', ')}]\n`;
  output += `Points/class: ${config.pointsPerClass}  lr: ${config.lr}  noise: ${config.noise}\n\n`;

  for (let iy = 0; iy < height; iy++) {
    let line = '';
    let currentColor = null;

    for (let ix = 0; ix < width; ix++) {
      const ch = gridChar[iy][ix];
      const colorCode = gridColor[iy][ix];

      if (noColor || colorCode == null) {
        line += ch;
      } else {
        if (colorCode !== currentColor) {
          line += ansi.reset + ansi.color(colorCode);
          currentColor = colorCode;
        }
        line += ch;
      }
    }

    if (!noColor && currentColor !== null) {
      line += ansi.reset;
    }
    output += line + '\n';
  }

  process.stdout.write(output);
}

// ======================= Main ========================================
function main() {
  const config = parseArgs();
  const rng = makeRNG(config.seed);

  const { data, labels } = generateSpiral(config.pointsPerClass, config.noise, rng);
  const bounds = computeBounds(data);
  const net = new Network(2, config.hidden, 2, rng);

  let interrupted = false;

  const cleanup = () => {
    process.stdout.write(ansi.reset + ansi.showCursor + '\n');
  };

  process.on('SIGINT', () => {
    interrupted = true;
  });
  process.on('exit', cleanup);

  process.stdout.write(ansi.hideCursor);

  const totalEpochs = config.epochs;
  for (let epoch = 1; epoch <= totalEpochs; epoch++) {
    if (interrupted) break;

    let totalLoss = 0;
    let correct = 0;

    const indices = Array.from({ length: data.length }, (_, i) => i);
    shuffleInPlace(indices, rng);

    for (let k = 0; k < indices.length; k++) {
      const idx = indices[k];
      const x = data[idx];
      const y = labels[idx];
      const { loss, correct: c } = net.trainSample(x, y, config.lr);
      totalLoss += loss;
      correct += c;
    }

    const avgLoss = totalLoss / data.length;
    const acc = correct / data.length;

    if (
      epoch === 1 ||
      epoch === totalEpochs ||
      epoch % config.refreshEvery === 0 ||
      interrupted
    ) {
      drawFrame(config, epoch, avgLoss, acc, net, data, labels, bounds);
    }
  }
}

main();
