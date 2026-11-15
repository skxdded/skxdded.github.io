/* worker.js: runs the heavy experiment loops off the main thread so the terminal stays responsive. */
importScripts('core.js');

var P = null;

function emit(m) { self.postMessage(m); }

self.onmessage = function (e) {
  var m = e.data;
  try {
    if (m.cmd === 'init') {
      P = Core.prep(m.data);
      emit({ type: 'ready' });
    } else if (!P) {
      emit({ type: 'error', text: 'No data loaded in the worker.' });
    } else if (m.cmd === 'baseline') {
      var t0 = Date.now(), b = Core.baseline(P);
      emit({ type: 'baseline', acc: b.acc, correct: b.correct, total: b.total, alpha: b.alpha, ms: Date.now() - t0 });
    } else if (m.cmd === 'experiment') {
      Core.runExperiment(P, m.cfg, emit);
    } else if (m.cmd === 'noise') {
      Core.runNoise(P, m.cfg, emit);
    }
  } catch (err) {
    emit({ type: 'error', text: String(err && err.message || err) });
  }
};
