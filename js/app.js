/* app.js: the terminal, the live charts and the inspector. The experiment itself lives in core.js / worker.js. */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var out = $('out'), inp = $('cmd'), badge = $('badge');
  var reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var sleep = function (ms) { return new Promise(function (r) { setTimeout(r, ms); }); };
  var clone = function (o) { return JSON.parse(JSON.stringify(o)); };
  var pct = function (x, d) { return (x * 100).toFixed(d == null ? 1 : d) + '%'; };

  var DEFAULTS = { sizes: [10, 50, 100, 300, 1000], seeds: 5, rho: 0.9, leak: 0.5, density: 0.1, inputScale: 1,
                   noiseSigmas: [0, 0.01, 0.05, 0.1, 0.3], noiseUnits: 300 };
  var cfg = clone(DEFAULTS);
  var data = null;           // raw dataset currently loaded
  var worker = null, busy = false;
  var results = fresh();
  function fresh() { return { base: null, rows: {}, order: [], noise: {}, norder: [], cfg: null, synthetic: false }; }

  // ------------------------------------------------------------ terminal output
  function ln(text, cls) {
    var d = document.createElement('div');
    d.className = 'ln' + (cls ? ' ' + cls : '');
    d.textContent = text;
    out.appendChild(d); out.scrollTop = out.scrollHeight;
    return d;
  }
  function typeLn(text, cls) {
    if (reduce) { ln(text, cls); return Promise.resolve(); }
    var d = ln('', cls), i = 0;
    return new Promise(function (res) {
      (function tick() {
        i = Math.min(text.length, i + 3);
        d.textContent = text.slice(0, i); out.scrollTop = out.scrollHeight;
        if (i < text.length) requestAnimationFrame(tick); else res();
      })();
    });
  }
  function echo(cmd) {
    var d = document.createElement('div'); d.className = 'ln';
    var p = document.createElement('span'); p.className = 'p'; p.textContent = 'reservoir@lab:~$ ';
    d.appendChild(p); d.appendChild(document.createTextNode(cmd));
    out.appendChild(d); out.scrollTop = out.scrollHeight;
  }
  function setBadge(text, cls) { badge.textContent = text; badge.className = 'badge ' + cls; }

  // ------------------------------------------------------------ worker
  function startWorker() {
    if (worker) worker.terminate();
    worker = new Worker('js/worker.js');
    worker.onmessage = onMsg;
    worker.onerror = function (e) { ln('Worker error: ' + (e.message || 'unknown'), 'err'); setBusy(false); };
    if (data) worker.postMessage({ cmd: 'init', data: data });
  }
  function setBusy(b) {
    busy = b;
    document.querySelectorAll('#chips .chip').forEach(function (c) {
      c.disabled = b && c.getAttribute('data-cmd') !== 'stop';
    });
  }
  var prog = null, done = 0, total = 0;
  function bar(n, t) {
    var w = 24, f = Math.round(w * n / t);
    return '[' + '█'.repeat(f) + '░'.repeat(w - f) + '] ' + n + '/' + t;
  }
  function onMsg(e) {
    var m = e.data;
    if (m.type === 'baseline') {
      results.base = m;
      ln('  baseline (no reservoir): ' + pct(m.acc) + '  (' + m.correct + '/' + m.total + ' correct)', 'warn');
      render();
    } else if (m.type === 'point') {
      var r = results.rows[m.n] || (results.rows[m.n] = { accs: [], done: false });
      r.accs.push(m.acc); done++;
      ln('  n=' + String(m.n).padEnd(5) + ' seed ' + (m.seed + 1) + '/' + m.seeds + '   acc ' + pct(m.acc) + '  (' + m.correct + '/' + m.total + ')   ' + (m.ms / 1000).toFixed(1) + 's', 'dim');
      if (prog) prog.textContent = bar(done, total);
      render();
    } else if (m.type === 'size_done') {
      var rr = results.rows[m.n]; rr.done = true; rr.mean = m.mean; rr.std = m.std;
      ln('→ ' + String(m.n).padEnd(5) + ' neurons: ' + pct(m.mean) + ' ± ' + pct(m.std), 'acc');
      render();
    } else if (m.type === 'npoint') {
      var q = results.noise[m.sigma] || (results.noise[m.sigma] = { accs: [], done: false });
      q.accs.push(m.acc); done++;
      if (prog) prog.textContent = bar(done, total);
      render();
    } else if (m.type === 'noise_done') {
      var qq = results.noise[m.sigma]; qq.done = true; qq.mean = m.mean; qq.std = m.std;
      ln('→ noise ' + String(m.sigma).padEnd(5) + ': ' + pct(m.mean) + ' ± ' + pct(m.std), 'acc');
      render();
    } else if (m.type === 'done') {
      ln('');
      ln('Done in ' + (m.ms / 1000).toFixed(1) + 's.', 'ok');
      verdict();
      setBusy(false);
    } else if (m.type === 'error') {
      ln('Error: ' + m.text, 'err'); setBusy(false);
    }
  }
  function verdict() {
    var best = null;
    results.order.forEach(function (n) { var r = results.rows[n]; if (r && r.done && (!best || r.mean > best.mean)) best = { n: n, mean: r.mean }; });
    if (results.base && best) {
      var diff = (best.mean - results.base.acc) * 100;
      ln('Baseline ' + pct(results.base.acc) + ' · best reservoir ' + pct(best.mean) + ' at n=' + best.n + ' (' + (diff >= 0 ? '+' : '') + diff.toFixed(1) + ' points).');
      ln('A gap this small can hide inside run-to-run noise, so read the spread, not just the average.', 'dim');
    }
    if (results.synthetic) ln('These are SYNTHETIC results: pipeline check only. Do not report them.', 'warn');
  }

  // ------------------------------------------------------------ data
  function setData(d) {
    data = d; results = fresh(); results.synthetic = !!d.synthetic;
    if (d.synthetic) setBadge('data: SYNTHETIC', 'warn'); else setBadge('data: Japanese Vowels ✓', 'ok');
    startWorker();
    ln('Loaded ' + d.source + ': ' + d.train.length + ' training / ' + d.test.length + ' test recordings, 12 channels, 9 speakers.', d.synthetic ? 'warn' : 'ok');
    render(); drawInspector();
  }
  async function tryAutoLoad() {
    try {
      var a = await fetch('data/ae.train'), b = await fetch('data/ae.test');
      if (!a.ok || !b.ok) throw new Error('missing');
      setData(Core.loadUCI(await a.text(), await b.text()));
      return true;
    } catch (err) {
      ln('Could not find data/ae.train and data/ae.test.', 'warn');
      ln('Press "Load data" and choose ae.train and ae.test (or the whole zip) from the Japanese Vowels download.', 'dim');
      ln('Or type: data synthetic   (made-up data, just to check the pipeline runs)', 'dim');
      return false;
    }
  }
  function loadJSZip() {
    return new Promise(function (res, rej) {
      if (window.JSZip) return res();
      var s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
      s.onload = res; s.onerror = function () { rej(new Error('Could not load the zip reader. Unzip the file and choose ae.train and ae.test instead.')); };
      document.head.appendChild(s);
    });
  }
  $('file').addEventListener('change', async function (e) {
    var files = Array.prototype.slice.call(e.target.files), tr = null, te = null;
    try {
      for (var i = 0; i < files.length; i++) {
        var f = files[i], nm = f.name.toLowerCase();
        if (nm.slice(-4) === '.zip') {
          await loadJSZip();
          var z = await window.JSZip.loadAsync(f);
          var ftr = z.file(/ae\.train$/)[0], fte = z.file(/ae\.test$/)[0];
          if (!ftr || !fte) throw new Error('That zip does not contain ae.train and ae.test.');
          tr = await ftr.async('string'); te = await fte.async('string');
        } else if (nm.indexOf('train') >= 0) tr = await f.text();
        else if (nm.indexOf('test') >= 0) te = await f.text();
      }
      if (!tr || !te) throw new Error('Choose both ae.train and ae.test (or the zip).');
      setData(Core.loadUCI(tr, te));
    } catch (err) { ln('Could not load: ' + err.message, 'err'); }
    e.target.value = '';
  });

  // ------------------------------------------------------------ commands
  var HELP = [
    ['help', 'show this list'],
    ['load', 'load the Japanese Vowels files (ae.train, ae.test)'],
    ['info', 'describe the loaded dataset'],
    ['baseline', 'accuracy with no reservoir at all'],
    ['run', 'full experiment: reservoir sizes x seeds (after a baseline)'],
    ['run quick', 'smaller, faster version (sizes 10/50/100, 2 seeds)'],
    ['noise', 'add noise inside the reservoir and watch accuracy drop'],
    ['config', 'show the current settings'],
    ['set <key> <value>', 'keys: seeds, units (e.g. 10,50,100), radius, leak, density, noiseunits'],
    ['reset', 'restore default settings'],
    ['stop', 'cancel a running experiment'],
    ['export', 'print the results as CSV'],
    ['data synthetic', 'use made-up data to check the pipeline (never report it)'],
    ['clear', 'clear the screen']
  ];
  var NAMES = ['help', 'load', 'info', 'baseline', 'run', 'noise', 'config', 'set', 'reset', 'stop', 'export', 'data', 'clear', 'ls', 'about'];

  function showConfig() {
    ln('  units      ' + cfg.sizes.join(', '));
    ln('  seeds      ' + cfg.seeds);
    ln('  radius     ' + cfg.rho + '   (spectral radius, connection strength)');
    ln('  leak       ' + cfg.leak);
    ln('  density    ' + cfg.density + '   (fraction of neurons connected)');
    ln('  noise      ' + cfg.noiseSigmas.join(', ') + ' at ' + cfg.noiseUnits + ' units');
  }
  function needData() {
    if (!data) { ln('No data loaded yet. Press "Load data", or type: data synthetic', 'warn'); return true; }
    return false;
  }
  function startRun(kind, quick) {
    if (needData()) return;
    if (busy) { ln('Already running. Type stop to cancel.', 'warn'); return; }
    var c = clone(cfg);
    if (quick) { c.sizes = [10, 50, 100]; c.seeds = 2; }
    done = 0;
    if (kind === 'run') {
      results = fresh(); results.synthetic = !!data.synthetic; results.cfg = c; results.order = c.sizes.slice();
      total = c.sizes.length * c.seeds;
      ln('Running ' + c.sizes.length + ' sizes × ' + c.seeds + ' seeds  (radius ' + c.rho + ', leak ' + c.leak + ', density ' + c.density + ')');
      if (c.sizes.indexOf(1000) >= 0 && !quick) ln('The 1000-neuron runs are the slow ones. Hang tight.', 'dim');
      worker.postMessage({ cmd: 'baseline' });
      prog = ln(bar(0, total), 'acc');
      worker.postMessage({ cmd: 'experiment', cfg: c });
    } else {
      results.noise = {}; results.norder = c.noiseSigmas.slice(); results.cfg = results.cfg || c; results.noiseCfg = c;
      total = c.noiseSigmas.length * c.seeds;
      ln('Noise test at ' + c.noiseUnits + ' neurons, ' + c.seeds + ' seeds each. A crude stand-in for a messy living substrate, not a model of biology.');
      prog = ln(bar(0, total), 'acc');
      worker.postMessage({ cmd: 'noise', cfg: c });
    }
    setBusy(true); render();
  }
  function exec(line) {
    line = line.trim(); if (!line) return;
    echo(line);
    var a = line.split(/\s+/), c = a[0].toLowerCase(), rest = a.slice(1);
    switch (c) {
      case 'help': ln('Commands:', 'bold'); HELP.forEach(function (h) { ln('  ' + h[0].padEnd(20) + h[1], 'dim'); }); break;
      case 'load': ln('Looking for data/ae.train and data/ae.test ...', 'dim');
        tryAutoLoad().then(function (ok) { if (!ok) $('file').click(); }); break;
      case 'info':
        if (needData()) break;
        ln('  source    ' + data.source); ln('  train     ' + data.train.length + ' recordings (30 per speaker)');
        ln('  test      ' + data.test.length + ' recordings'); ln('  channels  12 numbers per moment; recordings are 7 to 29 moments long');
        ln('  task      guess which of 9 speakers made each recording'); break;
      case 'baseline':
        if (needData()) break; if (busy) { ln('Already running. Type stop to cancel.', 'warn'); break; }
        ln('Training the readout on the raw sound numbers (no reservoir) ...', 'dim');
        setBusy(true); worker.postMessage({ cmd: 'baseline' });
        var h = function () { if (results.base) { setBusy(false); } else setTimeout(h, 100); }; setTimeout(h, 100); break;
      case 'run': startRun('run', rest[0] === 'quick' || rest[0] === '--quick'); break;
      case 'noise': startRun('noise', false); break;
      case 'config': showConfig(); break;
      case 'set': setKey(rest); break;
      case 'reset': cfg = clone(DEFAULTS); ln('Settings restored.', 'ok'); showConfig(); break;
      case 'stop':
        if (!busy) { ln('Nothing is running.', 'dim'); break; }
        startWorker(); setBusy(false); ln('Stopped.', 'warn'); break;
      case 'export': ln(csv()); break;
      case 'data':
        if (rest[0] === 'synthetic') { ln('Using SYNTHETIC data. It is made up, only to check the pipeline runs. Never report these numbers.', 'warn'); setData(Core.synthetic(7)); }
        else ln('Usage: data synthetic', 'dim'); break;
      case 'clear': out.innerHTML = ''; break;
      case 'ls': ln('index.html  css/  js/  data/  experiment/  README.md', 'acc'); break;
      case 'about': ln('Brainoware Reservoir Lab, by Moksha Chaudhari (Grade 11, Sinclair Secondary).'); ln('A fixed random reservoir plus a trained linear readout, after Cai et al. (2023).', 'dim'); break;
      default: ln('Command not found: ' + c + '. Type help.', 'err');
    }
  }
  function setKey(a) {
    var k = (a[0] || '').toLowerCase(), v = a.slice(1).join(' ').replace(/\s/g, ''), x;
    function bad(msg) { ln(msg, 'err'); }
    if (!k || !v) { bad('Usage: set <key> <value>   e.g. set radius 0.5'); return; }
    if (k === 'seeds') { x = parseInt(v, 10); if (!(x >= 1 && x <= 10)) return bad('seeds must be 1 to 10'); cfg.seeds = x; }
    else if (k === 'units' || k === 'sizes') {
      var list = v.split(',').map(Number).filter(function (n) { return n >= 2 && n <= 1000; });
      if (!list.length || list.length > 8) return bad('units: up to 8 numbers between 2 and 1000, e.g. 10,50,100');
      cfg.sizes = list.sort(function (p, q) { return p - q; });
    }
    else if (k === 'radius') { x = parseFloat(v); if (!(x >= 0.05 && x <= 1.5)) return bad('radius must be 0.05 to 1.5'); cfg.rho = x; }
    else if (k === 'leak') { x = parseFloat(v); if (!(x >= 0.05 && x <= 1)) return bad('leak must be 0.05 to 1'); cfg.leak = x; }
    else if (k === 'density') { x = parseFloat(v); if (!(x >= 0.01 && x <= 0.5)) return bad('density must be 0.01 to 0.5'); cfg.density = x; }
    else if (k === 'noiseunits') { x = parseInt(v, 10); if (!(x >= 10 && x <= 1000)) return bad('noiseunits must be 10 to 1000'); cfg.noiseUnits = x; }
    else return bad('Unknown key: ' + k);
    ln('OK. Now:', 'ok'); showConfig();
  }

  // history + tab completion
  var hist = [], hi = 0;
  inp.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { var v = inp.value; if (v.trim()) { hist.push(v); } hi = hist.length; inp.value = ''; exec(v); }
    else if (e.key === 'ArrowUp') { if (hi > 0) { hi--; inp.value = hist[hi]; } e.preventDefault(); }
    else if (e.key === 'ArrowDown') { if (hi < hist.length - 1) { hi++; inp.value = hist[hi]; } else { hi = hist.length; inp.value = ''; } e.preventDefault(); }
    else if (e.key === 'Tab') {
      e.preventDefault(); var t = inp.value.trim().toLowerCase(); if (!t) return;
      var m = NAMES.filter(function (n) { return n.indexOf(t) === 0; }); if (m.length === 1) inp.value = m[0] + ' ';
    } else if (e.key === 'l' && e.ctrlKey) { e.preventDefault(); out.innerHTML = ''; }
  });
  out.addEventListener('click', function () { if (!window.getSelection().toString()) inp.focus(); });
  document.querySelectorAll('#chips .chip').forEach(function (c) {
    c.addEventListener('click', function () { exec(c.getAttribute('data-cmd')); });
  });

  // ------------------------------------------------------------ results panel
  function csv() {
    var L = ['setup,value,mean_accuracy,std_accuracy,seeds'];
    if (results.base) L.push('baseline,none,' + results.base.acc.toFixed(4) + ',,1');
    results.order.forEach(function (n) { var r = results.rows[n]; if (r && r.done) L.push('reservoir_units,' + n + ',' + r.mean.toFixed(4) + ',' + r.std.toFixed(4) + ',' + r.accs.length); });
    results.norder.forEach(function (s) { var r = results.noise[s]; if (r && r.done) L.push('noise_sigma_at_' + ((results.noiseCfg || cfg).noiseUnits) + '_units,' + s + ',' + r.mean.toFixed(4) + ',' + r.std.toFixed(4) + ',' + r.accs.length); });
    return L.join('\n');
  }
  $('dl').addEventListener('click', function () {
    var blob = new Blob([csv()], { type: 'text/csv' }), a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'results.csv'; document.body.appendChild(a); a.click(); a.remove();
  });
  function chartSVG(o) {
    var W = 560, H = 300, L = 46, R = 14, T = 14, B = 44, pw = W - L - R, ph = H - T - B, pad = 20, i;
    var vals = [];
    o.dots.forEach(function (d) { vals.push(d.y); });
    o.line.forEach(function (p) { vals.push(p.y - p.s, p.y + p.s); });
    if (o.base != null) vals.push(o.base);
    var lo = Math.max(0, Math.floor((Math.min.apply(null, vals) - 3) / 2) * 2), hiV = 100;
    if (hiV - lo < 8) lo = hiV - 8;
    function Y(v) { return T + (hiV - v) / (hiV - lo) * ph; }
    var n = o.ticks.length;
    function X(k) {
      var frac;
      if (o.log) { var a = Math.log10(o.ticks[0].v), b = Math.log10(o.ticks[n - 1].v); frac = n > 1 ? (Math.log10(k) - a) / (b - a) : 0.5; }
      else frac = n > 1 ? k / (n - 1) : 0.5;
      return L + pad + frac * (pw - 2 * pad);
    }
    var step = (hiV - lo) > 24 ? 5 : 2, s = '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="' + o.aria + '">';
    for (var v = Math.ceil(lo / step) * step; v <= hiV; v += step) s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '" stroke="var(--line)"/><text x="' + (L - 8) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + v + '</text>';
    o.ticks.forEach(function (t, idx) { var x = X(o.log ? t.v : idx); s += '<text x="' + x + '" y="' + (H - B + 18) + '" text-anchor="middle">' + t.label + '</text>'; });
    s += '<text x="' + (L + pw / 2) + '" y="' + (H - 6) + '" text-anchor="middle">' + o.xLabel + '</text><text transform="translate(12 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle">test accuracy (%)</text>';
    if (o.base != null) s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(o.base) + '" y2="' + Y(o.base) + '" stroke="var(--warm)" stroke-width="2" stroke-dasharray="6 5"/><text x="' + (W - R - 4) + '" y="' + (Y(o.base) + 15) + '" text-anchor="end" style="fill:var(--warm)">baseline ' + o.base.toFixed(1) + '%</text>';
    o.dots.forEach(function (d) { s += '<circle cx="' + X(d.k) + '" cy="' + Y(d.y) + '" r="3.2" fill="none" stroke="var(--accent)" stroke-opacity="0.55"/>'; });
    var path = ''; o.line.forEach(function (p, idx) { path += (idx ? 'L' : 'M') + X(p.k) + ' ' + Y(p.y); });
    if (o.line.length > 1) s += '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2.4"/>';
    o.line.forEach(function (p) {
      var x = X(p.k), a = Math.max(lo, p.y - p.s), b = Math.min(hiV, p.y + p.s);
      s += '<line x1="' + x + '" x2="' + x + '" y1="' + Y(a) + '" y2="' + Y(b) + '" stroke="var(--accent)" stroke-width="2"/><line x1="' + (x - 5) + '" x2="' + (x + 5) + '" y1="' + Y(a) + '" y2="' + Y(a) + '" stroke="var(--accent)" stroke-width="2"/><line x1="' + (x - 5) + '" x2="' + (x + 5) + '" y1="' + Y(b) + '" y2="' + Y(b) + '" stroke="var(--accent)" stroke-width="2"/>';
      s += '<circle cx="' + x + '" cy="' + Y(p.y) + '" r="5" fill="var(--accent)"><title>' + p.t + '</title></circle>';
      var ly = Y(b) - 8; if (ly < T + 8) ly = Y(a) + 18;
      s += '<text x="' + x + '" y="' + ly + '" text-anchor="middle" style="fill:var(--fg)">' + p.y.toFixed(1) + '%</text>';
    });
    return s + '</svg>';
  }
  var EMPTY = '<div class="empty">Run it in the terminal and this fills in live.</div>';
  function render() {
    // stats
    var best = null, st = '';
    results.order.forEach(function (n) { var r = results.rows[n]; if (r && r.done && (!best || r.mean > best.mean)) best = { n: n, mean: r.mean, std: r.std }; });
    if (results.base) st += '<div class="stat base"><div class="num">' + pct(results.base.acc) + '</div><div class="lab">Baseline, no reservoir</div></div>';
    if (best) st += '<div class="stat best"><div class="num">' + pct(best.mean) + '</div><div class="lab">Best reservoir: ' + best.n + ' neurons (± ' + pct(best.std) + ')</div></div>';
    if (best && results.base) { var dd = (best.mean - results.base.acc) * 100; st += '<div class="stat"><div class="num">' + (dd >= 0 ? '+' : '') + dd.toFixed(1) + '</div><div class="lab">Points above the baseline</div></div>'; }
    $('stats').innerHTML = st;
    $('synthBanner').className = 'banner' + (results.synthetic ? ' show' : '');
    // size chart
    var ticks = results.order.map(function (n) { return { v: n, label: String(n) }; }), dots = [], line = [];
    results.order.forEach(function (n) {
      var r = results.rows[n]; if (!r) return;
      r.accs.forEach(function (a) { dots.push({ k: n, y: a * 100 }); });
      if (r.done) line.push({ k: n, y: r.mean * 100, s: r.std * 100, t: n + ' neurons: ' + pct(r.mean) + ' ± ' + pct(r.std) });
    });
    if (!dots.length && !results.base) $('chart').innerHTML = EMPTY;
    else if (!ticks.length) $('chart').innerHTML = EMPTY;
    else $('chart').innerHTML = chartSVG({ ticks: ticks, log: true, dots: dots, line: line, base: results.base ? results.base.acc * 100 : null, xLabel: 'reservoir size (neurons)', aria: 'Test accuracy against reservoir size' });
    // noise chart
    var nt = results.norder.map(function (s) { return { v: s, label: String(s) }; }), nd = [], nl = [];
    results.norder.forEach(function (s, idx) {
      var r = results.noise[s]; if (!r) return;
      r.accs.forEach(function (a) { nd.push({ k: idx, y: a * 100 }); });
      if (r.done) nl.push({ k: idx, y: r.mean * 100, s: r.std * 100, t: 'noise ' + s + ': ' + pct(r.mean) });
    });
    $('nchart').innerHTML = nt.length && nd.length ? chartSVG({ ticks: nt, log: false, dots: nd, line: nl, base: null, xLabel: 'noise strength inside the reservoir', aria: 'Test accuracy against noise strength' }) : EMPTY;
    // table
    var tb = '';
    if (results.base) tb += '<tr class="base"><td>No reservoir (baseline)</td><td>' + pct(results.base.acc) + '</td><td>one run</td><td>1</td></tr>';
    results.order.forEach(function (n) { var r = results.rows[n]; if (r && r.done) tb += '<tr><td>' + n + ' neurons</td><td>' + pct(r.mean) + '</td><td>± ' + pct(r.std) + '</td><td>' + r.accs.length + '</td></tr>'; });
    results.norder.forEach(function (s) { var r = results.noise[s]; if (r && r.done) tb += '<tr><td>' + ((results.noiseCfg || cfg).noiseUnits) + ' neurons, noise ' + s + '</td><td>' + pct(r.mean) + '</td><td>± ' + pct(r.std) + '</td><td>' + r.accs.length + '</td></tr>'; });
    $('rows').innerHTML = tb || '<tr><td colspan="4" class="muted">Nothing yet.</td></tr>';
    var c = results.cfg; $('cfgline').textContent = c ? 'radius ' + c.rho + ' · leak ' + c.leak + ' · density ' + c.density + ' · ' + c.seeds + ' seeds' : '';
  }

  // ------------------------------------------------------------ inspector (runs in the page, small reservoirs only)
  var insp = { seed: 1, P: null, src: null };
  function hexRGB(v) { v = v.trim(); if (v[0] !== '#') return [128, 128, 128]; if (v.length === 4) v = '#' + v[1] + v[1] + v[2] + v[2] + v[3] + v[3]; return [parseInt(v.substr(1, 2), 16), parseInt(v.substr(3, 2), 16), parseInt(v.substr(5, 2), 16)]; }
  function tok(n) { return getComputedStyle(document.documentElement).getPropertyValue(n); }
  function mix(a, b, t) { return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * t) + ',' + Math.round(a[1] + (b[1] - a[1]) * t) + ',' + Math.round(a[2] + (b[2] - a[2]) * t) + ')'; }
  function sized(c, h) { var d = window.devicePixelRatio || 1, w = c.clientWidth || 300; c.width = Math.round(w * d); c.height = Math.round(h * d); c.style.height = h + 'px'; var g = c.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0); return { g: g, w: w, h: h }; }
  function panel(p, R, P, real) {
    var sp = +$(p + 'S').value, rec = +$(p + 'R').value, idx = sp * 30 + rec - 1, seq = P.train[idx];
    $(p + 'Ro').textContent = rec;
    var res = Core.features(R, seq, 0.5, 0, Core.rng(9), true), H = res.hist, T = H.length, n = R.n;
    $(p + 'Cap').textContent = 'Speaker ' + (sp + 1) + ' · recording ' + rec + ' · ' + T + ' moments · ' + (real ? 'real data' : 'SYNTHETIC data');
    var surf = hexRGB(tok('--surface')), acc = hexRGB(tok('--accent')), warm = hexRGB(tok('--warm')), line = tok('--line').trim();
    var A = sized($(p + 'In'), 90), g = A.g, c, t, i;
    g.strokeStyle = line; g.beginPath(); g.moveTo(0, A.h / 2); g.lineTo(A.w, A.h / 2); g.stroke();
    for (c = 0; c < 12; c++) {
      g.strokeStyle = mix(acc, warm, c / 11); g.globalAlpha = 0.75; g.lineWidth = 1.3; g.beginPath();
      for (t = 0; t < T; t++) { var x = T > 1 ? t / (T - 1) * (A.w - 2) + 1 : A.w / 2, y = A.h / 2 - seq[t][c] * (A.h / 2 - 5); t ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
    }
    g.globalAlpha = 1;
    var M = sized($(p + 'Map'), 150), gm = M.g, cw = M.w / T, rh = M.h / n;
    for (i = 0; i < n; i++) for (t = 0; t < T; t++) {
      var v = Math.max(-1, Math.min(1, H[t][i] * 1.8)); gm.fillStyle = v >= 0 ? mix(surf, acc, v) : mix(surf, warm, -v); gm.fillRect(t * cw, i * rh, cw + 0.6, rh + 0.6);
    }
    var V = sized($(p + 'Avg'), 16), gv = V.g, vw = V.w / n;
    for (i = 0; i < n; i++) { var m = Math.max(-1, Math.min(1, res.f[i] * 2.6)); gv.fillStyle = m >= 0 ? mix(surf, acc, m) : mix(surf, warm, -m); gv.fillRect(i * vw, 0, vw + 0.6, V.h); }
  }
  function drawInspector() {
    var real = !!(data && !data.synthetic);
    if (!insp.P || insp.src !== data) { insp.P = Core.prep(data || Core.synthetic(7)); insp.src = data; }
    var n = +$('inN').value; $('inNo').textContent = n;
    $('inSource').textContent = real ? 'Showing real recordings from the training set.' : (data ? 'Showing SYNTHETIC recordings (pipeline check).' : 'Showing SYNTHETIC recordings until you load the real data.');
    var R = Core.makeReservoir(n, { rho: 0.9, density: 0.1, inputScale: 1 }, insp.seed);
    panel('a', R, insp.P, real); panel('b', R, insp.P, real);
  }
  ['aS', 'bS'].forEach(function (id, k) {
    var s = $(id); for (var i = 0; i < 9; i++) { var o = document.createElement('option'); o.value = i; o.textContent = 'Speaker ' + (i + 1); s.appendChild(o); }
    s.value = k === 0 ? 0 : 4;
  });
  ['inN', 'aS', 'bS', 'aR', 'bR'].forEach(function (id) { $(id).addEventListener('input', drawInspector); });
  $('inSeed').addEventListener('click', function () { insp.seed++; drawInspector(); });
  var rt; window.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(drawInspector, 150); });
  if (window.matchMedia) { var mq = window.matchMedia('(prefers-color-scheme: dark)'); if (mq.addEventListener) mq.addEventListener('change', drawInspector); }
  new MutationObserver(drawInspector).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ------------------------------------------------------------ start
  async function boot() {
    render(); drawInspector(); startWorker();
    await typeLn('Brainoware Reservoir Lab', 'bold');
    await typeLn('A fixed random reservoir + a trained readout, rebuilt from a brain-organoid speech study.', 'dim');
    ln('');
    await typeLn('Type help for commands, or use the buttons below.', 'dim');
    ln('');
    await tryAutoLoad();
    ln('');
  }
  boot();
})();
