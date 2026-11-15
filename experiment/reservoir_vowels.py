"""
Brainoware-style reservoir computing on speech, with a SIMULATED reservoir.

Idea (from Cai et al., Nature Electronics 2023, "Brainoware"):
  speech features -> fixed, untrained "reservoir" -> trained linear readout
In Brainoware the reservoir is a living brain organoid on an electrode array.
Here the reservoir is a random recurrent network (an echo state network) I
simulate in numpy. Only the readout is trained, exactly like in Brainoware.

Dataset: UCI "Japanese Vowels" (9 speakers say /ae/; each utterance is a
sequence of 12 audio coefficients). Brainoware used a similar speaker-
recognition task, but NOT the identical split, so my numbers are not a
head-to-head comparison with the paper's ~78%.

Run:
    pip install numpy scikit-learn matplotlib
    python reservoir_vowels.py            # real data (downloads from UCI)
    python reservoir_vowels.py --synthetic   # pipeline check only, fake data
"""
import argparse, io, os, re, sys, urllib.request, zipfile
import numpy as np
from sklearn.linear_model import RidgeClassifierCV

URL = "https://archive.ics.uci.edu/static/public/128/japanese+vowels.zip"
TEST_COUNTS = [31, 35, 88, 44, 29, 24, 40, 50, 29]   # utterances per speaker in ae.test
TRAIN_PER_SPEAKER = 30


# ---------------------------------------------------------------- data
def parse_blocks(text):
    """Each utterance = block of lines with 12 numbers; blocks split by blank lines."""
    blocks = re.split(r"\n\s*\n", text.strip())
    return [np.array([[float(v) for v in ln.split()] for ln in b.strip().splitlines()])
            for b in blocks if b.strip()]


def load_real(cache="data"):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "japanese_vowels.zip")
    if not os.path.exists(path):
        print("Downloading dataset from UCI ...")
        urllib.request.urlretrieve(URL, path)
    with zipfile.ZipFile(path) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        train = parse_blocks(z.read(names["ae.train"]).decode())
        test = parse_blocks(z.read(names["ae.test"]).decode())
    ytr = np.repeat(np.arange(9), TRAIN_PER_SPEAKER)
    yte = np.concatenate([[i] * c for i, c in enumerate(TEST_COUNTS)])
    assert len(train) == len(ytr) == 270 and len(test) == len(yte) == 370, \
        f"unexpected sizes: {len(train)}, {len(test)}"
    return train, ytr, test, yte


def load_synthetic(seed=0):
    """FAKE data (9 noisy 'speakers'), only to check that the code runs."""
    rng = np.random.default_rng(seed)
    protos = rng.normal(size=(9, 12, 3))
    def make(n_each):
        X, y = [], []
        for c in range(9):
            for _ in range(n_each):
                T = rng.integers(8, 28)
                t = np.linspace(0, 1, T)[:, None]
                seq = protos[c, :, 0] + protos[c, :, 1] * t + protos[c, :, 2] * t ** 2
                X.append(seq + rng.normal(scale=0.8, size=seq.shape)); y.append(c)
        return X, np.array(y)
    Xtr, ytr = make(30); Xte, yte = make(40)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------- reservoir
class Reservoir:
    """x_t = (1-a) x_{t-1} + a * tanh(W_in u_t + W x_{t-1}); W and W_in are random and FIXED."""
    def __init__(self, n_in, n_units, spectral_radius=0.9, input_scale=1.0,
                 leak=0.5, density=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-input_scale, input_scale, size=(n_units, n_in))
        while True:   # small reservoirs can come out all zeros, so redraw until W is usable
            W = rng.normal(size=(n_units, n_units)) * (rng.random((n_units, n_units)) < density)
            radius = max(abs(np.linalg.eigvals(W)))
            if radius > 1e-6:
                break
        self.W = W * (spectral_radius / radius)      # rescale so the echo-state property holds
        self.leak, self.n = leak, n_units

    def features(self, seq):
        x = np.zeros(self.n); states = []
        for u in seq:
            x = (1 - self.leak) * x + self.leak * np.tanh(self.W_in @ u + self.W @ x)
            states.append(x)
        S = np.array(states)
        return np.concatenate([S.mean(0), S[-1]])    # average state + final state


def scale_fit(train):
    allx = np.vstack(train)
    return allx.min(0), allx.max(0)

def scale_apply(seqs, lo, hi):
    return [2 * (s - lo) / (hi - lo + 1e-9) - 1 for s in seqs]


def raw_features(seq):
    """Baseline with NO reservoir: just average + final frame of the raw coefficients."""
    return np.concatenate([seq.mean(0), seq[-1]])


def accuracy(Ftr, ytr, Fte, yte):
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 3, 15))   # the only trained part
    clf.fit(Ftr, ytr)
    return (clf.predict(Fte) == yte).mean()


# ---------------------------------------------------------------- experiments
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noise", action="store_true", help="stretch goal: noisy reservoir")
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load_synthetic() if args.synthetic else load_real()
    if args.synthetic:
        print("*** SYNTHETIC DATA: pipeline check only, do NOT report these numbers ***")
    lo, hi = scale_fit(Xtr)
    Xtr, Xte = scale_apply(Xtr, lo, hi), scale_apply(Xte, lo, hi)
    n_in = Xtr[0].shape[1]

    base = accuracy(np.array([raw_features(s) for s in Xtr]), ytr,
                    np.array([raw_features(s) for s in Xte]), yte)
    print(f"\nBaseline, no reservoir (linear readout on raw features): {base:.1%}")

    sizes = [10, 50, 100, 300, 1000]
    rows = []
    print("\nReservoir size vs test accuracy (mean +/- std over seeds):")
    for n in sizes:
        accs = []
        for seed in range(args.seeds):
            r = Reservoir(n_in, n, seed=seed)
            Ftr = np.array([r.features(s) for s in Xtr])
            Fte = np.array([r.features(s) for s in Xte])
            accs.append(accuracy(Ftr, ytr, Fte, yte))
        rows.append((n, np.mean(accs), np.std(accs)))
        print(f"  {n:>5} units: {np.mean(accs):.1%} +/- {np.std(accs):.1%}")

    noise_rows = []
    if args.noise:
        print("\nStretch: add noise INSIDE the reservoir (a crude stand-in for a living substrate)")
        for sigma in [0.0, 0.01, 0.05, 0.1, 0.3]:
            accs = []
            for seed in range(args.seeds):
                r = Reservoir(n_in, 300, seed=seed)
                rng = np.random.default_rng(100 + seed)
                def noisy(seq):
                    x = np.zeros(r.n); S = []
                    for u in seq:
                        x = (1 - r.leak) * x + r.leak * np.tanh(r.W_in @ u + r.W @ x) \
                            + rng.normal(scale=sigma, size=r.n)
                        S.append(x)
                    S = np.array(S); return np.concatenate([S.mean(0), S[-1]])
                accs.append(accuracy(np.array([noisy(s) for s in Xtr]), ytr,
                                     np.array([noisy(s) for s in Xte]), yte))
            noise_rows.append((sigma, np.mean(accs), np.std(accs)))
            print(f"  noise {sigma:<5}: {np.mean(accs):.1%} +/- {np.std(accs):.1%}")

    # save + plot
    import csv, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tag = "synthetic_" if args.synthetic else ""
    with open(f"{tag}results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["reservoir_units", "mean_acc", "std_acc"]); w.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar([r[0] for r in rows], [r[1] * 100 for r in rows],
                yerr=[r[2] * 100 for r in rows], marker="o", capsize=4, label="Simulated reservoir + linear readout")
    ax.axhline(base * 100, color="gray", ls="--", label="No reservoir (raw features)")
    ax.set_xscale("log"); ax.set_xlabel("Reservoir size (units)"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Speaker recognition with a simulated reservoir"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{tag}accuracy_vs_size.png", dpi=200)
    print(f"\nSaved {tag}results.csv and {tag}accuracy_vs_size.png")


if __name__ == "__main__":
    main()
