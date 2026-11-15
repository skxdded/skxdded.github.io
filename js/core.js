/*
 * core.js: the experiment engine. No dependencies.
 * Same method as experiment/reservoir_vowels.py:
 *   scaled speech -> fixed random reservoir (echo state network) -> mean + final state
 *   -> trained ridge classifier readout (alpha chosen by leave-one-out) -> speaker.
 * Runs in a Web Worker (see worker.js) and in the page (inspector) and in Node (tests).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Core = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var NIN = 12;
  var TEST_COUNTS = [31, 35, 88, 44, 29, 24, 40, 50, 29]; // utterances per speaker in ae.test
  var TRAIN_PER = 30;                                       // utterances per speaker in ae.train

  // ---------------------------------------------------------------- random numbers
  function rng(seed) {
    var s = seed | 0;
    return function () {
      s = (s + 0x6D2B79F5) | 0;
      var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    var u = 0, v;
    while (u === 0) u = r();
    v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---------------------------------------------------------------- data
  // UCI format: each utterance is a block of lines (12 numbers per line), blocks split by blank lines.
  function parseBlocks(text) {
    text = text.replace(/\r/g, '').trim();
    return text.split(/\n[ \t]*\n/).filter(function (b) { return b.trim(); }).map(function (b) {
      return b.trim().split('\n').map(function (l) {
        return Float64Array.from(l.trim().split(/\s+/).map(Number));
      });
    });
  }
  function labels(counts) {
    var y = [], k, i;
    for (k = 0; k < counts.length; k++) for (i = 0; i < counts[k]; i++) y.push(k);
    return y;
  }
  function loadUCI(trainText, testText) {
    var tr = parseBlocks(trainText), te = parseBlocks(testText);
    if (tr.length !== 270 || te.length !== 370) {
      throw new Error('Expected 270 train and 370 test utterances, got ' + tr.length + ' and ' + te.length +
        '. Are these the ae.train and ae.test files from the Japanese Vowels zip?');
    }
    return {
      train: tr, test: te,
      ytr: labels(TEST_COUNTS.map(function () { return TRAIN_PER; })),
      yte: labels(TEST_COUNTS),
      source: 'UCI Japanese Vowels', synthetic: false
    };
  }
  // FAKE data for a pipeline check only. Never report numbers from it.
  function synthetic(seed) {
    var r = rng(seed || 7), protos = [], k, c, j;
    for (k = 0; k < 9; k++) { var p = []; for (c = 0; c < NIN; c++) p.push([gauss(r), gauss(r), gauss(r)]); protos.push(p); }
    function make(counts) {
      var X = [], y = [], i, t;
      counts.forEach(function (cnt, cls) {
        for (i = 0; i < cnt; i++) {
          var T = 7 + Math.floor(r() * 21), seq = [];
          for (t = 0; t < T; t++) {
            var row = new Float64Array(NIN), tt = t / (T - 1);
            for (j = 0; j < NIN; j++) row[j] = protos[cls][j][0] + protos[cls][j][1] * tt + protos[cls][j][2] * tt * tt + 1.6 * gauss(r);
            seq.push(row);
          }
          X.push(seq); y.push(cls);
        }
      });
      return { X: X, y: y };
    }
    var a = make(TEST_COUNTS.map(function () { return TRAIN_PER; })), b = make(TEST_COUNTS);
    return { train: a.X, ytr: a.y, test: b.X, yte: b.y, source: 'SYNTHETIC (pipeline check only)', synthetic: true };
  }
  // Scale every channel to [-1, 1] using TRAINING data only, so the test set cannot leak in.
  function prep(data) {
    var lo = new Float64Array(NIN).fill(Infinity), hi = new Float64Array(NIN).fill(-Infinity);
    data.train.forEach(function (seq) { seq.forEach(function (row) {
      for (var c = 0; c < NIN; c++) { if (row[c] < lo[c]) lo[c] = row[c]; if (row[c] > hi[c]) hi[c] = row[c]; }
    }); });
    function sc(seq) { return seq.map(function (row) {
      var o = new Float64Array(NIN); for (var c = 0; c < NIN; c++) o[c] = 2 * (row[c] - lo[c]) / (hi[c] - lo[c] + 1e-9) - 1; return o; }); }
    return { train: data.train.map(sc), test: data.test.map(sc), ytr: data.ytr, yte: data.yte, synthetic: !!data.synthetic };
  }

  // ---------------------------------------------------------------- reservoir
  // x_t = (1 - leak) * x_{t-1} + leak * tanh(W_in u_t + W x_{t-1}) (+ noise). W_in and W are random and never trained.
  function specRad(rowptr, col, val, n) {
    var v = new Float64Array(n), w = new Float64Array(n), i, p, k, s = 0, cnt = 0;
    for (i = 0; i < n; i++) v[i] = 1 + 0.01 * i;
    for (k = 0; k < 220; k++) {
      var nm = 0;
      for (i = 0; i < n; i++) { var a = 0; for (p = rowptr[i]; p < rowptr[i + 1]; p++) a += val[p] * v[col[p]]; w[i] = a; nm += a * a; }
      nm = Math.sqrt(nm);
      if (nm < 1e-12) return 0;
      for (i = 0; i < n; i++) v[i] = w[i] / nm;
      if (k >= 70) { s += Math.log(nm); cnt++; }
    }
    return Math.exp(s / cnt);
  }
  function makeReservoir(n, opts, seed) {
    var r = rng(seed), i, j, c, tries = 0, rowptr, col, val, rad;
    var Win = new Float64Array(n * NIN);
    for (i = 0; i < Win.length; i++) Win[i] = (r() * 2 - 1) * opts.inputScale;
    do { // tiny reservoirs can come out with no connections at all, so redraw until usable
      rowptr = new Int32Array(n + 1); var cc = [], vv = [];
      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) if (r() < opts.density) { cc.push(j); vv.push(gauss(r)); }
        rowptr[i + 1] = cc.length;
      }
      col = Int32Array.from(cc); val = Float64Array.from(vv);
      rad = specRad(rowptr, col, val, n);
      tries++;
    } while (rad < 1e-6 && tries < 300);
    var sc = rad > 1e-6 ? opts.rho / rad : 0;
    for (c = 0; c < val.length; c++) val[c] *= sc;
    return { n: n, Win: Win, rowptr: rowptr, col: col, val: val };
  }
  // Runs one utterance. Returns [mean state, final state] joined, and the full state history if asked.
  function features(R, seq, leak, noise, r, keep) {
    var n = R.n, x = new Float64Array(n), nx = new Float64Array(n), sum = new Float64Array(n), t, i, p, c, hist = keep ? [] : null;
    for (t = 0; t < seq.length; t++) {
      var u = seq[t];
      for (i = 0; i < n; i++) {
        var a = 0;
        for (c = 0; c < NIN; c++) a += R.Win[i * NIN + c] * u[c];
        for (p = R.rowptr[i]; p < R.rowptr[i + 1]; p++) a += R.val[p] * x[R.col[p]];
        nx[i] = (1 - leak) * x[i] + leak * Math.tanh(a) + (noise ? noise * gauss(r) : 0);
      }
      var tmp = x; x = nx; nx = tmp;
      for (i = 0; i < n; i++) sum[i] += x[i];
      if (keep) hist.push(Float64Array.from(x));
    }
    var f = new Float64Array(2 * n);
    for (i = 0; i < n; i++) { f[i] = sum[i] / seq.length; f[n + i] = x[i]; }
    return keep ? { f: f, hist: hist } : f;
  }
  function rawFeatures(seq) { // baseline: no reservoir, just average + last frame of the raw numbers
    var f = new Float64Array(2 * NIN), t, c;
    for (t = 0; t < seq.length; t++) for (c = 0; c < NIN; c++) f[c] += seq[t][c] / seq.length;
    for (c = 0; c < NIN; c++) f[NIN + c] = seq[seq.length - 1][c];
    return f;
  }

  // ---------------------------------------------------------------- linear algebra
  // Symmetric eigen-decomposition (Householder + implicit QL, after JAMA). V starts as the matrix, ends as eigenvectors.
  function eigSym(A) {
    var n = A.length, V = A.map(function (r) { return Float64Array.from(r); }), d = new Float64Array(n), e = new Float64Array(n);
    var i, j, k, f, g, h, hh, scale;
    for (j = 0; j < n; j++) d[j] = V[n - 1][j];
    for (i = n - 1; i > 0; i--) {
      scale = 0; h = 0;
      for (k = 0; k < i; k++) scale += Math.abs(d[k]);
      if (scale === 0) {
        e[i] = d[i - 1];
        for (j = 0; j < i; j++) { d[j] = V[i - 1][j]; V[i][j] = 0; V[j][i] = 0; }
      } else {
        for (k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
        f = d[i - 1]; g = Math.sqrt(h); if (f > 0) g = -g;
        e[i] = scale * g; h -= f * g; d[i - 1] = f - g;
        for (j = 0; j < i; j++) e[j] = 0;
        for (j = 0; j < i; j++) {
          f = d[j]; V[j][i] = f; g = e[j] + V[j][j] * f;
          for (k = j + 1; k <= i - 1; k++) { g += V[k][j] * d[k]; e[k] += V[k][j] * f; }
          e[j] = g;
        }
        f = 0;
        for (j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
        hh = f / (h + h);
        for (j = 0; j < i; j++) e[j] -= hh * d[j];
        for (j = 0; j < i; j++) {
          f = d[j]; g = e[j];
          for (k = j; k <= i - 1; k++) V[k][j] -= (f * e[k] + g * d[k]);
          d[j] = V[i - 1][j]; V[i][j] = 0;
        }
      }
      d[i] = h;
    }
    for (i = 0; i < n - 1; i++) {
      V[n - 1][i] = V[i][i]; V[i][i] = 1; h = d[i + 1];
      if (h !== 0) {
        for (k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
        for (j = 0; j <= i; j++) {
          g = 0;
          for (k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
          for (k = 0; k <= i; k++) V[k][j] -= g * d[k];
        }
      }
      for (k = 0; k <= i; k++) V[k][i + 1] = 0;
    }
    for (j = 0; j < n; j++) { d[j] = V[n - 1][j]; V[n - 1][j] = 0; }
    V[n - 1][n - 1] = 1; e[0] = 0;
    // QL iterations
    for (i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0;
    var tst1 = 0, eps = Math.pow(2, -52), l, m, iter, p, r, c, c2, c3, el1, s, s2, dl1;
    f = 0;
    for (l = 0; l < n; l++) {
      tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
      m = l;
      while (m < n) { if (Math.abs(e[m]) <= eps * tst1) break; m++; }
      if (m > l) {
        iter = 0;
        do {
          iter++;
          g = d[l]; p = (d[l + 1] - g) / (2 * e[l]); r = Math.hypot(p, 1); if (p < 0) r = -r;
          d[l] = e[l] / (p + r); d[l + 1] = e[l] * (p + r); dl1 = d[l + 1]; h = g - d[l];
          for (i = l + 2; i < n; i++) d[i] -= h;
          f += h; p = d[m]; c = 1; c2 = c; c3 = c; el1 = e[l + 1]; s = 0; s2 = 0;
          for (i = m - 1; i >= l; i--) {
            c3 = c2; c2 = c; s2 = s;
            g = c * e[i]; h = c * p; r = Math.hypot(p, e[i]);
            e[i + 1] = s * r; s = e[i] / r; c = p / r;
            p = c * d[i] - s * g; d[i + 1] = h + s * (c * g + s * d[i]);
            for (k = 0; k < n; k++) {
              h = V[k][i + 1]; V[k][i + 1] = s * V[k][i] + c * h; V[k][i] = c * V[k][i] - s * h;
            }
          }
          p = -s * s2 * c3 * el1 * e[l] / dl1; e[l] = s * p; d[l] = c * p;
        } while (Math.abs(e[l]) > eps * tst1 && iter < 200);
      }
      d[l] += f; e[l] = 0;
    }
    return { values: d, vectors: V };
  }

  // Ridge classifier with the regularisation strength picked by leave-one-out error (like sklearn's RidgeClassifierCV).
  // Works in the "dual" form: with 270 training utterances and up to 2000 features this is far cheaper.
  var ALPHAS = []; for (var q = 0; q < 15; q++) ALPHAS.push(Math.pow(10, -4 + 7 * q / 14));
  function ridgeAccuracy(Ftr, ytr, Fte, yte) {
    var m = Ftr.length, d = Ftr[0].length, K = 9, i, j, k, t;
    var mean = new Float64Array(d);
    for (i = 0; i < m; i++) for (j = 0; j < d; j++) mean[j] += Ftr[i][j] / m;
    var Fc = Ftr.map(function (f) { var o = new Float64Array(d); for (var j2 = 0; j2 < d; j2++) o[j2] = f[j2] - mean[j2]; return o; });
    var G = []; for (i = 0; i < m; i++) G.push(new Float64Array(m));
    for (i = 0; i < m; i++) for (j = i; j < m; j++) { var s = 0, a = Fc[i], b = Fc[j]; for (t = 0; t < d; t++) s += a[t] * b[t]; G[i][j] = s; G[j][i] = s; }
    var Y = [], ybar = new Float64Array(K);
    for (i = 0; i < m; i++) { var row = new Float64Array(K); for (k = 0; k < K; k++) { row[k] = ytr[i] === k ? 1 : -1; ybar[k] += row[k] / m; } Y.push(row); }
    for (i = 0; i < m; i++) for (k = 0; k < K; k++) Y[i][k] -= ybar[k];
    var eg = eigSym(G), lam = eg.values, U = eg.vectors;
    var UtY = []; for (j = 0; j < m; j++) { var ro = new Float64Array(K); for (i = 0; i < m; i++) for (k = 0; k < K; k++) ro[k] += U[i][j] * Y[i][k]; UtY.push(ro); }
    var bestA = ALPHAS[0], bestE = Infinity, bestC = null;
    ALPHAS.forEach(function (al) {
      var inv = new Float64Array(m); for (var jj = 0; jj < m; jj++) inv[jj] = 1 / (Math.max(lam[jj], 0) + al);
      var err = 0, C = [];
      for (var ii = 0; ii < m; ii++) {
        var gii = 0, c = new Float64Array(K);
        for (var jj2 = 0; jj2 < m; jj2++) { var w = U[ii][jj2] * inv[jj2]; gii += w * U[ii][jj2]; for (var kk = 0; kk < K; kk++) c[kk] += w * UtY[jj2][kk]; }
        for (var kk2 = 0; kk2 < K; kk2++) { var e = c[kk2] / gii; err += e * e; }
        C.push(c);
      }
      if (err < bestE) { bestE = err; bestA = al; bestC = C; }
    });
    var correct = 0, z = new Float64Array(d);
    for (i = 0; i < Fte.length; i++) {
      for (j = 0; j < d; j++) z[j] = Fte[i][j] - mean[j];
      var sc = Float64Array.from(ybar);
      for (t = 0; t < m; t++) { var dot = 0, ft = Fc[t]; for (j = 0; j < d; j++) dot += z[j] * ft[j]; for (k = 0; k < K; k++) sc[k] += bestC[t][k] * dot; }
      var arg = 0; for (k = 1; k < K; k++) if (sc[k] > sc[arg]) arg = k;
      if (arg === yte[i]) correct++;
    }
    return { acc: correct / Fte.length, alpha: bestA, correct: correct, total: Fte.length };
  }

  // ---------------------------------------------------------------- experiments
  function mean(a) { var s = 0; a.forEach(function (v) { s += v; }); return s / a.length; }
  function std(a) { var m = mean(a), s = 0; a.forEach(function (v) { s += (v - m) * (v - m); }); return Math.sqrt(s / a.length); }

  function baseline(P) {
    var res = ridgeAccuracy(P.train.map(rawFeatures), P.ytr, P.test.map(rawFeatures), P.yte);
    return res;
  }
  function evalOne(P, cfg, n, seed, sigma) {
    var R = makeReservoir(n, { rho: cfg.rho, density: cfg.density, inputScale: cfg.inputScale }, seed);
    var r = rng(100 + seed);
    var Ftr = P.train.map(function (s) { return features(R, s, cfg.leak, sigma, r); });
    var Fte = P.test.map(function (s) { return features(R, s, cfg.leak, sigma, r); });
    return ridgeAccuracy(Ftr, P.ytr, Fte, P.yte);
  }
  // emit({type, ...}) reports progress; stopped() lets a caller cancel between runs.
  function runExperiment(P, cfg, emit) {
    var t0 = Date.now();
    cfg.sizes.forEach(function (n) {
      var accs = [];
      for (var s = 0; s < cfg.seeds; s++) {
        var t1 = Date.now(), res = evalOne(P, cfg, n, s, 0);
        accs.push(res.acc);
        emit({ type: 'point', n: n, seed: s, seeds: cfg.seeds, acc: res.acc, correct: res.correct, total: res.total, alpha: res.alpha, ms: Date.now() - t1 });
      }
      emit({ type: 'size_done', n: n, mean: mean(accs), std: std(accs), accs: accs });
    });
    emit({ type: 'done', ms: Date.now() - t0 });
  }
  function runNoise(P, cfg, emit) {
    var t0 = Date.now();
    cfg.noiseSigmas.forEach(function (sg) {
      var accs = [];
      for (var s = 0; s < cfg.seeds; s++) {
        var res = evalOne(P, cfg, cfg.noiseUnits, s, sg);
        accs.push(res.acc);
        emit({ type: 'npoint', sigma: sg, seed: s, seeds: cfg.seeds, acc: res.acc });
      }
      emit({ type: 'noise_done', sigma: sg, mean: mean(accs), std: std(accs), accs: accs });
    });
    emit({ type: 'done', ms: Date.now() - t0 });
  }

  return {
    NIN: NIN, TEST_COUNTS: TEST_COUNTS, TRAIN_PER: TRAIN_PER,
    rng: rng, gauss: gauss, parseBlocks: parseBlocks, loadUCI: loadUCI, synthetic: synthetic, prep: prep,
    makeReservoir: makeReservoir, features: features, rawFeatures: rawFeatures, eigSym: eigSym,
    ridgeAccuracy: ridgeAccuracy, baseline: baseline, evalOne: evalOne, runExperiment: runExperiment, runNoise: runNoise,
    specRad: specRad, mean: mean, std: std
  };
});
